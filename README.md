
# Pupil Labs Recording Fixer

This is a utility script to fix recordings with corrupted video due to app crash
manifesting in unplayable videos due to missing moov atom or muxing issue resulting in
header error.

## Install

1. Install Python 3.10 or higher
2. ```bash
   $ pip install -e git+https://github.com/pupil-labs/pl-recover-recording.git#egg=pl-recover-recording
   ```

## Usage

```bash

$ pl-recover-recording --help

 Usage: pl-recover-recording [OPTIONS] REC_PATH

 Recover a recording

╭─ Arguments ──────────────────────────────────────────────────────────────────────────╮
│ *    rec_path      PATH  [default: None] [required]                                  │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────╮
│ --cleanup-temp-files    --no-cleanup-temp-files      [default: cleanup-temp-files]   │
│ --help                                               Show this message and exit.     │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

### Development

### Building Untrunc

It is important to use the 3.3.9 version of ffmpeg when building untrunc since the newer
versions 4+ do not work.

```bash
sudo apt install build-essentials yasm wget git
git clone https://github.com/anthwlock/untrunc
cd untrunc
make FF_VER=3.3.9
sudo cp untrunc /usr/local/bin
```
### Reference Video Creation

To recover video requires a known good reference video that is used to untruncate the
broken video. Reference videos have already been created for Neon recordings.

#### Neon Reference Video

##### Scene

- Scene reference video is created by muxing a pi recording audio track (41,000hz)
  with a neon video track

##### Sensor

- Sensor reference video is created by taking a regular Neon Sensor Module video

