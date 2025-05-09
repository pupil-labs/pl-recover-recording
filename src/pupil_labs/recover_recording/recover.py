"""

## Dependencies

# PYTHON
pip install \
    av \
    typer \
    structlog \
    numpy \
    rich

# FFMPEG
apt install ffmpeg

# UNTRUNC
sudo apt-get install yasm wget git build-essential
git clone https://github.com/anthwlock/untrunc
cd untrunc
make FF_VER=3.3.9
sudo cp untrunc /usr/local/bin

"""

import enum
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

import av
import av.error
import av.video
import numpy as np
import structlog
import typer
from rich.progress import Progress, track

PathLike = Path | str

APP_PATH = Path(__file__).resolve().parent
REFERENCE_VIDEOS_DIRECTORY = APP_PATH / "reference-videos"
RECOVERED_TEMP_FILES_DIRECTORY_NAME = "pl_recover_tmp_files"
HW_VS_SW_DELTA_THRESHOLD_SECONDS = 0.5
MAX_RECORDING_DURATION_NANOSECS = 7 * 60 * 60 * 1e9  # 7 hours
structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(sys.stderr),
)

logger = structlog.get_logger()


class NoVideoStreamException(Exception): ...


class DEFAULTS:
    resize_video: bool = True


class VideoKind(enum.Enum):
    NEON_SENSOR_MODULE = enum.auto()
    NEON_SCENE_CAMERA = enum.auto()
    PI_WORLD_CAMERA = enum.auto()
    PI_EYE_CAMERA_MP4 = enum.auto()
    PI_EYE_CAMERA_MJPEG = enum.auto()


VIDEO_KIND_FILE_PATTERNS = {
    VideoKind.NEON_SCENE_CAMERA: re.compile(r"^Neon Scene Camera v1 ps(\d+)\.mp4$"),
    VideoKind.NEON_SENSOR_MODULE: re.compile(r"^Neon Sensor Module v1 ps(\d+)\.mp4$"),
    VideoKind.PI_WORLD_CAMERA: re.compile(r"^PI world v1 ps(\d+)\.mp4$"),
    VideoKind.PI_EYE_CAMERA_MP4: re.compile(r"^PI (left|right) v1 ps(\d+)\.mp4$"),
}

REFERENCE_VIDEO_PATHS = {
    VideoKind.NEON_SCENE_CAMERA: REFERENCE_VIDEOS_DIRECTORY / "neon-scene-ref.mp4",
    VideoKind.NEON_SENSOR_MODULE: REFERENCE_VIDEOS_DIRECTORY / "neon-sensor-ref.mp4",
    VideoKind.PI_WORLD_CAMERA: REFERENCE_VIDEOS_DIRECTORY / "pi-world-ref.mp4",
    VideoKind.PI_EYE_CAMERA_MP4: REFERENCE_VIDEOS_DIRECTORY / "pi-eye-ref.mp4",
}
EXPECTED_VIDEO_RESOLUTIONS = {
    VideoKind.NEON_SCENE_CAMERA: (1600, 1200),
    VideoKind.NEON_SENSOR_MODULE: (384, 192),
    VideoKind.PI_WORLD_CAMERA: (1088, 1080),
    VideoKind.PI_EYE_CAMERA_MP4: (192, 192),
}
VIDEO_KINDS_WITH_AUDIO = {VideoKind.NEON_SCENE_CAMERA, VideoKind.PI_WORLD_CAMERA}
JSON_KIND_FILE_PATTERN = re.compile(r"^(info|template|wearer)\.json$")
TIME_KIND_FILE_PATTERN = re.compile(r"^.+\.time_?(hw|aux)?$")


class CameraCalibrationV2(object):
    """Class for converting v2 neon camera intrinsics/extrinsics from binary/json format"""

    dtype = np.dtype(
        [
            ("version", "u1"),
            ("serial", "6a"),
            ("scene_camera_matrix", "(3,3)d"),
            ("scene_distortion_coefficients", "8d"),
            ("scene_extrinsics_affine_matrix", "(4,4)d"),
            ("right_camera_matrix", "(3,3)d"),
            ("right_distortion_coefficients", "8d"),
            ("right_extrinsics_affine_matrix", "(4,4)d"),
            ("left_camera_matrix", "(3,3)d"),
            ("left_distortion_coefficients", "8d"),
            ("left_extrinsics_affine_matrix", "(4,4)d"),
            ("crc", "u4"),
        ]
    )

    @classmethod
    def binary_to_json(cls, bytes_object):
        calibration_data = np.frombuffer(
            bytes_object,
            cls.dtype,
        )[0]
        result = {
            name: calibration_data[name].tolist()
            for name in calibration_data.dtype.names
        }
        result["serial"] = calibration_data["serial"].decode("utf8")
        return result


def get_container_error(video_file_path):
    av.logging.set_level(av.logging.INFO)
    with av.logging.Capture() as av_logs:
        error = ""
        try:
            c = av.open(str(video_file_path))
            next(c.decode(video=0))
        except av.error.InvalidDataError as e:
            error_str = "?"
            if e.log:
                error_str = " - ".join([str(part) for part in e.log])
            error = f"corrupt video: {error_str}"
        except Exception as e:
            error = f"error: {e.__class__.__name__} {av_logs}"
        if error:
            return error


def run_command(args: list[str]):
    args = [str(arg) for arg in args]
    shell_command = shlex.join(args)

    logger.debug("running command", cmd=shell_command)
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(process.stdout, newline=""):
        if "\n" in line:
            logger.debug(line.rstrip("\n"))
        else:
            print(line, end="")

    process.wait()
    error_code = process.poll()
    if error_code:
        raise RuntimeError(f"error running command: {shell_command}")


def untrunc_video(
    bad_video_path: PathLike,
    good_video_path: PathLike,
    fixed_video_path: PathLike,
):
    return run_command(
        [
            "untrunc",
            "-dst",
            str(fixed_video_path),
            str(good_video_path),
            str(bad_video_path),
        ]
    )


def remux_video_with_timestamps(
    video_path: PathLike,
    timestamps_unix_time_nanosecs: list[int],
    output_file_path: PathLike,
):
    input_container = av.open(str(video_path))
    input_container.streams.video[0].thread_type = "AUTO"
    input_video_stream = input_container.streams.video[0]

    offsets_from_zero_in_nanosecs = (
        timestamps_unix_time_nanosecs - timestamps_unix_time_nanosecs[0]
    )
    frame_stream = track(
        zip(offsets_from_zero_in_nanosecs, input_container.demux(video=0)),
        total=input_video_stream.frames,
        description="remuxing video file",
    )
    with av.open(str(output_file_path), "w") as output_container:
        output_video_stream = output_container.add_stream(template=input_video_stream)
        previous_pts = None
        for offset_in_nanoseconds, packet in frame_stream:
            if packet.pts is None:
                continue
            new_pts = int(round(offset_in_nanoseconds / 1e9 / packet.time_base))
            if previous_pts is not None and previous_pts == new_pts:
                new_pts += 1
            previous_pts = new_pts
            packet.dts = packet.pts = new_pts
            packet.stream = output_video_stream
            output_container.mux_one(packet)


def resize_video(
    video_path: PathLike,
    output_path: PathLike,
    resolution: tuple[int, int],
):
    stream = av.open(str(video_path)).streams.video[0]
    stream_resolution = (stream.width, stream.height)
    if stream_resolution == resolution:
        logger.warning("video already at resolution", resolution=resolution)
        return
    bitrate = f"{round(stream.bit_rate / 1e6, 1)}M"
    width, height = resolution
    ffmpeg_resize_args = [
        "ffmpeg",
        "-vsync",
        "vfr",
        "-i",
        str(video_path),
        "-b:v",
        bitrate,
        "-r",
        "90000",
        "-vf",
        f"scale={width}:{height}",
        str(output_path),
        "-y",
    ]
    return run_command(ffmpeg_resize_args)


def combine_video_audio(
    video_path: PathLike, audio_path: PathLike, output_path: PathLike
):
    ffmpeg_combine_video_audio_args = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c",
        "copy",
        str(output_path),
        "-y",
    ]
    return run_command(ffmpeg_combine_video_audio_args)


HTTPPath = str


@dataclass
class RecordingVideoFixer:
    path: Path | HTTPPath | str
    logger: structlog.BoundLogger = logger

    def __post_init__(self):
        if str(self.path).startswith("http"):
            self.path = HTTPPath(self.path)
        else:
            self.path = Path(self.path)

    @property
    def filename(self):
        if isinstance(self.path, Path):
            return self.path.name
        parsed = urlparse(self.path)
        return unquote(parsed.path.split("/")[-1])

    @property
    def kind(self) -> VideoKind | None:
        for video_kind, pattern in VIDEO_KIND_FILE_PATTERNS.items():
            if pattern.findall(self.filename):
                return video_kind
        return None

    @property
    def error(self):
        container_error = get_container_error(self.path)
        if container_error:
            return container_error

        if len(av.open(str(self.path)).streams.video) == 0:
            return NoVideoStreamException()
        if self.expected_resolution != self.video_resolution:
            return f"invalid resolution: {'x'.join(map(str, self.video_resolution))}"

    @property
    def expected_resolution(self):
        return EXPECTED_VIDEO_RESOLUTIONS[self.kind]

    @property
    def video_resolution(self):
        try:
            stream = av.open(str(self.path)).streams.video[0]
            return stream.width, stream.height
        except Exception as e:
            self.logger.warning(
                "could not load resolution", path=self.path, error=str(e)
            )

    @property
    def temp_files(self):
        temp_files = self.path.parent / RECOVERED_TEMP_FILES_DIRECTORY_NAME
        temp_files.mkdir(exist_ok=True)
        return temp_files

    @property
    def paths(self):
        class Paths:
            original = self.path
            time_file = self.path.with_suffix(".time")
            backup = self.path.with_suffix(".mp4.original.mp4")

            untrunced = self.temp_files / self.path.with_suffix(".untrunced.mp4").name
            untrunced_audio = (
                self.temp_files / self.path.with_suffix(".untrunced_audio.aac").name
            )
            untrunced_video = (
                self.temp_files / self.path.with_suffix(".untrunced_video.mp4").name
            )
            remuxed_video = (
                self.temp_files / self.path.with_suffix(".remuxed_video.mp4").name
            )
            resized = self.temp_files / self.path.with_suffix(".resized.mp4").name
            fixed = self.temp_files / self.path.with_suffix(".fixed.mp4").name

        paths = Paths()
        return paths

    @property
    def reference_video_path(self):
        video_path = REFERENCE_VIDEO_PATHS.get(self.kind)
        if not video_path:
            raise RuntimeError(f"no reference video for kind: {self.kind}")
        return video_path

    def extract_untrunced_audio(self):
        ffmpeg_extract_audio_args = [
            "ffmpeg",
            "-i",
            self.paths.untrunced,
            "-c",
            "copy",
            self.paths.untrunced_audio,
            "-y",
        ]
        return run_command(ffmpeg_extract_audio_args)

    def extract_untrunced_video(self):
        ffmpeg_extract_video_args = [
            "ffmpeg",
            "-i",
            self.paths.untrunced,
            "-an",
            "-c",
            "copy",
            self.paths.untrunced_video,
            "-y",
        ]
        return run_command(ffmpeg_extract_video_args)

    @property
    def timestamps_from_time_file(self):
        return np.fromfile(self.paths.time_file, "<u8")

    def _check_and_fix_corrupt_video(self):
        error_message = get_container_error(self.path)
        if not error_message:
            return

        if "corrupt video" not in error_message:
            self.logger.warning(
                "not an error we can fix", path=self.path, error=error_message
            )
            return

        self.logger.warning(
            "error in container detected", path=self.path, error=error_message
        )
        if "header" in error_message or "moov" in error_message:
            self.logger.info(
                "untruncating broken video", path=self.path, error=error_message
            )
            untrunc_video(self.path, self.reference_video_path, self.paths.untrunced)
            if self.kind in VIDEO_KINDS_WITH_AUDIO:
                self.extract_untrunced_audio()
            self.extract_untrunced_video()

            remux_video_with_timestamps(
                self.paths.untrunced_video,
                self.timestamps_from_time_file,
                self.paths.remuxed_video,
            )

            if self.kind in VIDEO_KINDS_WITH_AUDIO:
                self.logger.info("combining audio and video tracks", path=self.path)
                logger.info("combining video/audio tracks")
                combine_video_audio(
                    self.paths.remuxed_video,
                    self.paths.untrunced_audio,
                    self.paths.fixed,
                )
            else:
                shutil.copy(self.paths.remuxed_video, self.paths.fixed)

            if not self.paths.backup.exists():
                logger.info(
                    "backing up file",
                    original=self.paths.original,
                    backup=self.paths.backup,
                )
                shutil.move(self.paths.original, self.paths.backup)

            logger.debug(
                "replacing original with recovered",
                original=self.paths.original,
            )
            shutil.move(self.paths.fixed, self.paths.original)

    def recover(self):
        if not isinstance(self.path, Path):
            raise Exception("can only recover files on disk")

        if self.path.stat().st_size < 500000:
            self.logger.warning("video too small", path=self.path)
            return
        self._check_and_fix_corrupt_video()
        self._check_and_fix_resolution()

    def _check_and_fix_resolution(self):
        if self.video_resolution is None:
            self.logger.warning("can not get resolution for video", path=self.path)
            return

        if self.video_resolution != self.expected_resolution:
            resize_video(
                self.paths.original,
                self.paths.resized,
                self.expected_resolution,
            )
            if not self.paths.backup.exists():
                logger.info(
                    "backing up file",
                    original=self.paths.original,
                    backup=self.paths.backup,
                )
                shutil.move(self.paths.original, self.paths.backup)

            logger.debug(
                "replacing original with recovered",
                original=self.paths.original,
            )
            shutil.move(self.paths.resized, self.paths.original)

    def __repr__(self):
        return f"<RecordingVideo({self.path})>"


@dataclass
class RecordingFixer:
    rec_path: Path
    cleanup_temp_files: bool = True

    @property
    def logger(self):
        return logger

    @property
    def progress(self):
        return Progress()

    @property
    def temp_file_path(self):
        path = self.rec_path / RECOVERED_TEMP_FILES_DIRECTORY_NAME
        return path

    def _recover_json_file(self, file_path: Path):
        assert file_path.suffix == ".json"
        issues = []

        json_bytes = file_path.read_bytes()
        try:
            json.loads(json_bytes)
        except json.JSONDecodeError as decode_error:
            logger.warning("json file had error", path=file_path, error=decode_error)
            issues.append(f"{file_path} had invalid json: {str(decode_error)}")

            # remove invalid part of json
            maybe_valid_part = json_bytes[: decode_error.pos]
            try:
                fixed_json = json.loads(maybe_valid_part)
            except json.JSONDecodeError:
                logger.error(
                    "json file has error and is not recoverable", path=file_path
                )
            else:
                new_json_bytes = json.dumps(
                    fixed_json, indent=2, sort_keys=True
                ).encode("UTF-8")
                if isinstance(fixed_json, dict):
                    logger.info(
                        "json file has error and is recoverable", path=file_path
                    )
                    backup_file_path = file_path.with_suffix(".json.original.json")
                    if not backup_file_path.exists():
                        logger.info("backing up json", path=backup_file_path)
                        shutil.move(file_path, backup_file_path)
                    (self.rec_path / file_path.name).write_bytes(new_json_bytes)
                else:
                    logger.error(
                        "json file has error and is not recoverable", path=file_path
                    )
        else:
            logger.info("json has no error, skipping", path=file_path)

        return issues

    def _find_max_timestamp_from_time_files(self):
        max_timestamp = 0
        for file in self.rec_path.glob("*.time"):
            if "original" in file.name:
                continue
            timestamps = np.fromfile(file, "<u8")
            max_timestamp = max(max_timestamp, timestamps.max())
        return int(max_timestamp)

    def _process_info_json(self):
        issues = []
        info_json_file = self.rec_path / "info.json"
        info_json_backup_file = self.rec_path / "info.json.original.json"
        info = json.loads(info_json_file.read_bytes())

        duration = info.get("duration")
        if not duration or duration > MAX_RECORDING_DURATION_NANOSECS:
            logger.warning(
                f"info.json had invalid duration",
                duration=duration,
                path=info_json_file,
            )
            issues.append(f"info.json had invalid duration: {duration}")
            max_timestamp = self._find_max_timestamp_from_time_files()
            potential_duration = max_timestamp - info["start_time"]

            if potential_duration < MAX_RECORDING_DURATION_NANOSECS:
                if not info_json_backup_file.exists():
                    logger.info("backing up info.json", path=info_json_backup_file)
                    shutil.move(info_json_file, info_json_backup_file)

                info["duration"] = int(potential_duration)
                new_json_bytes = json.dumps(info, indent=2, sort_keys=True).encode(
                    "UTF-8"
                )
                info_json_file.write_bytes(new_json_bytes)

        # recover module_serial_number of Neon recording's info.json
        module_serial_number = info.get("module_serial_number")
        not_pi = "scene_camera_serial_number" not in info

        if not module_serial_number and not_pi:
            logger.warning("fixing Neon's module_serial_number in info.json")
            calibration_bin_file = self.rec_path / "calibration.bin"
            try:
                # this exact message is being used for sending alert
                issues.append(f"missing module_serial_number")
                # read Neon's calibration file
                data = CameraCalibrationV2.binary_to_json(
                    calibration_bin_file.read_bytes()
                )
                info["module_serial_number"] = data["serial"]
                new_json_bytes = json.dumps(info, indent=2, sort_keys=True).encode(
                    "UTF-8"
                )
                info_json_file.write_bytes(new_json_bytes)
            except Exception as err:
                logger.error(
                    f"error fixing Neon's module_serial_number in info.json: {err}"
                )
                issues.append(
                    f"failed to fix missing module_serial_number in info.json"
                )

        return issues

    def _process_json_files(self):
        logger.info("checking json files")

        issues = []
        for json_file_path in self.rec_path.glob("*.json"):
            if "original" in json_file_path.name:
                continue
            if not JSON_KIND_FILE_PATTERN.findall(json_file_path.name):
                continue
            logger.debug("checking json file", path=json_file_path)
            issues.extend(self._recover_json_file(json_file_path))
        return issues

    def _process_video_files(self, resize_video: bool = DEFAULTS.resize_video):
        issues = []
        logger.info("checking corrupt video files")
        for file_path in self.rec_path.glob("*.mp4"):
            video_file = RecordingVideoFixer(file_path, logger=self.logger)
            if not video_file.kind:
                continue
            logger.debug("checking video file", path=file_path)
            error = video_file.error
            if error:
                issues.append(f"{video_file.path} had container error: {error}")
                logger.warning("video has error", error=error, path=video_file.path)
                video_file.recover()

        return issues

    def _process_event_files(self):
        logger.info("checking event files")
        issues = []

        info_json_path = self.rec_path / "info.json"
        event_text_path = self.rec_path / "event.txt"
        event_time_path = self.rec_path / "event.time"

        if not event_time_path.exists():
            issues.append("missing event.time missing")
            logger.warning("event.time missing, creating")
            event_time_path.touch()
        if not event_text_path.exists():
            issues.append("missing event.txt missing")
            logger.warning("event.txt missing, creating")
            event_text_path.touch()

        info = json.load(open(info_json_path, "rb"))

        event_timestamps = np.fromfile(event_time_path, "<u8")
        event_names = (
            open(event_text_path, "r").read().splitlines()[: len(event_timestamps)]
        )

        missing_recording_begin = False
        missing_recording_end = False
        if not event_names or event_names[0] != "recording.begin":
            missing_recording_begin = True
        if not event_names or event_names[-1] != "recording.end":
            missing_recording_end = True

        if missing_recording_begin:
            logger.warning("missing recording.begin event")
            event_names = ["recording.begin"] + event_names
            event_timestamps = np.concatenate([[info["start_time"]], event_timestamps])
            issues.append("missing recording.begin in event.txt")

        if missing_recording_end:
            logger.warning("missing recording.end event")
            event_names = event_names + ["recording.end"]
            rec_end_time = info["start_time"] + info["duration"]
            event_timestamps = np.concatenate([event_timestamps, [rec_end_time]])
            issues.append("missing recording.end in event.txt")

        if missing_recording_begin or missing_recording_end:
            backup_event_time_path = event_time_path.with_suffix(".time.original.time")
            if not backup_event_time_path.exists():
                logger.info("backing up event.time", path=backup_event_time_path)
                shutil.move(event_time_path, backup_event_time_path)

            logger.info("writing new event.time file")
            np.array(event_timestamps, "<u8").tofile(event_time_path)

            backup_event_text_path = event_time_path.with_suffix(".txt.original.txt")
            if not backup_event_text_path.exists():
                logger.info("backing up event.txt", path=backup_event_text_path)
                shutil.move(event_text_path, backup_event_text_path)

            logger.info("writing new event.txt file")
            event_text_path.write_text("\n".join(event_names) + "\n", "utf8")

        return issues

    def _process_time_files(self):
        issues = []
        file_paths = [
            file_path
            for file_path in self.rec_path.glob("*.time*")
            if TIME_KIND_FILE_PATTERN.findall(file_path.name)
        ]

        for file_path in file_paths:
            if "original" in file_path.name:
                continue

            if file_path.suffix == ".time_aux":
                time_file_path = file_path.with_suffix(".time")
                time_hw_file_path = file_path.with_suffix(".time_hw")
                time_backup_file_path = file_path.with_suffix(".time.original.time")
                time_aux_file_path = file_path
                time_aux_backup_file_path = file_path.with_suffix(
                    ".time_aux.original.time_aux"
                )

                if (
                    not time_file_path
                    or not time_aux_file_path.stat().st_size
                    or not time_file_path.stat().st_size
                ):
                    continue

                hw_ts = np.fromfile(time_file_path, "<i8")
                if not np.all(np.diff(hw_ts) > 0):
                    logger.warning(
                        f"hw timestamps not monotonic",
                        path=time_file_path,
                    )
                    issues.append(f"hw timestamps not monotonic for {time_file_path}")
                    if not time_backup_file_path.exists():
                        shutil.copy(time_file_path, time_backup_file_path)
                    hw_ts.sort()
                    hw_ts.tofile(time_file_path, format="<u8")

                sw_ts = np.fromfile(time_aux_file_path, "<i8")
                if not np.all(np.diff(sw_ts) > 0):
                    logger.warning(
                        f"sw timestamps not monotonic",
                        path=time_aux_file_path,
                    )
                    issues.append(
                        f"sw timestamps not monotonic for {time_aux_file_path}"
                    )
                    if not time_aux_backup_file_path.exists():
                        shutil.copy(time_aux_file_path, time_aux_backup_file_path)
                    sw_ts.sort()
                    sw_ts.tofile(time_aux_file_path, format="<u8")

                if time_hw_file_path not in file_paths:
                    max_delta_secs = np.abs((hw_ts - sw_ts)).max() / 1e9
                    if max_delta_secs > HW_VS_SW_DELTA_THRESHOLD_SECONDS:
                        logger.warning(
                            "hw timestamps seem incorrect",
                            threshold=HW_VS_SW_DELTA_THRESHOLD_SECONDS,
                            diff_secs=max_delta_secs,
                            hw=time_file_path,
                            sw=time_aux_file_path,
                        )
                        issues.append(
                            f"sw and hw timestamps differ for {time_file_path}"
                        )
                        logger.info("copying .time to .time_hw", path=time_file_path)
                        shutil.copy(time_file_path, time_hw_file_path)
                        logger.warning(
                            "replacing .time with sw timestamps", path=time_file_path
                        )
                        if not time_backup_file_path.exists():
                            shutil.copy(time_file_path, time_backup_file_path)
                        shutil.move(time_aux_file_path, time_file_path)
        return issues

    def process(self, resize_video: bool = DEFAULTS.resize_video):
        issues = []

        # must run in order since some depend on previous files to be correct
        try:
            issues.extend(self._process_json_files())
            issues.extend(self._process_time_files())
            issues.extend(self._process_info_json())
            issues.extend(self._process_event_files())
            issues.extend(self._process_video_files(resize_video=resize_video))
        finally:
            if self.cleanup_temp_files and self.temp_file_path.exists():
                logger.warning(
                    "deleting pl_recover temp files", path=self.temp_file_path
                )
                if self.temp_file_path.exists():
                    shutil.rmtree(self.temp_file_path)

        for issue in issues:
            logger.info(f"issue found: {issue}")
        return issues


cli = typer.Typer(help="Recording Fixer", no_args_is_help=True)


@cli.command()
def recover_recording(
    rec_path: Path,
    cleanup_temp_files: bool = True,
    resize_video: Annotated[
        bool, typer.Argument(help="Resize video")
    ] = DEFAULTS.resize_video,
):
    """
    Recover a recording
    """
    logger.info("fixing recording path", path=rec_path)
    fixer = RecordingFixer(rec_path, cleanup_temp_files=cleanup_temp_files)
    errors = fixer.process(resize_video=resize_video)
    return errors


if __name__ == "__main__":
    cli()
