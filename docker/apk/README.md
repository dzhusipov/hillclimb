# HCR2 APK Files

Place Hill Climb Racing 2 split APK files here. They are gitignored.

## How to get the APK

1. Go to [APKMirror](https://www.apkmirror.com/apk/fingersoft/hill-climb-racing-2/)
2. Download the latest version (choose **APK** variant, not Bundle)
3. If it's a split APK (XAPK/APKS), extract it to get individual `.apk` files
4. Place all `.apk` files in this directory

## Expected files

Typical split APK structure:

```
base.apk                    # Main application
split_config.arm64_v8a.apk  # ARM64 native libraries (for physical devices)
split_config.x86_64.apk     # x86_64 native libs (for ReDroid emulator)
split_config.xxhdpi.apk     # Resources for screen density
```

For ReDroid (x86_64 emulator), you need `base.apk` + `split_config.x86_64.apk` + density split.

## Install

```bash
# Install on all running emulators
../scripts/manage.sh install-apk

# Or manually on a specific emulator
adb -s localhost:5555 install-multiple *.apk
```

## Notes

- If the game crashes on ReDroid, it may need ARM translation (libhoudini).
  Switch to a ReDroid image with native bridge support.
- If the game requires Google Play Services, use a GApps image
  (`redroid/redroid:14.0.0-gapps`) or MicroG.
