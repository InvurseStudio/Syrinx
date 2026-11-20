#!/bin/zsh
# Builds the Syrinx app icon from assets/Syrinx.icon (an Icon Composer document).
# Needs Xcode 26 or later. The source shapes live in Vireo's assets/brand.
#
#   ./make-icons.sh mac <dir>   Assets.car + Syrinx.icns for Syrinx.app
#   ./make-icons.sh windows     ../../windows/SyrinxManager/assets/Syrinx.ico (committed)
set -e

cd "$(dirname "$0")"

ICON_DOC="assets/Syrinx.icon"
ICTOOL="$(xcode-select -p)/../Applications/Icon Composer.app/Contents/Executables/ictool"

if [[ ! -x "$ICTOOL" ]] || ! xcrun --find actool >/dev/null 2>&1; then
    echo "Xcode 26 or later is required to build the app icon." >&2
    exit 1
fi

# Renders the macOS icon at $1 px, centered on a transparent $2 px canvas.
render_mac() {
    local body=$1 canvas=$2 out=$3
    "$ICTOOL" "$ICON_DOC" --export-image --output-file "$out" --platform macOS \
        --rendition Default --width "$body" --height "$body" --scale 1 >/dev/null
    if [[ "$body" != "$canvas" ]]; then
        sips --padToHeightWidth "$canvas" "$canvas" "$out" >/dev/null
    fi
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

case "$1" in
    mac)
        out="$2"
        [[ -n "$out" ]] || { echo "Usage: $0 mac <output dir>" >&2; exit 1; }
        mkdir -p "$out"

        # Assets.car carries the layered icon for macOS 26+ (light, dark, clear
        # and tinted), and CFBundleIconName points at it.
        xcrun actool "$ICON_DOC" --compile "$work" --platform macosx \
            --minimum-deployment-target 14.0 --app-icon Syrinx \
            --output-partial-info-plist "$work/partial.plist" >/dev/null
        cp "$work/Assets.car" "$out/Assets.car"

        # actool's .icns fallback for macOS 14 and 15 stops at 256 px, so add
        # the large sizes. The body is 824 of 1024 px, Apple's icon grid.
        iconutil -c iconset "$work/Syrinx.icns" -o "$work/Syrinx.iconset"
        for size in 256 512; do
            render_mac $((size * 824 / 1024)) $size "$work/Syrinx.iconset/icon_${size}x${size}.png"
            render_mac $((size * 2 * 824 / 1024)) $((size * 2)) "$work/Syrinx.iconset/icon_${size}x${size}@2x.png"
        done
        iconutil -c icns "$work/Syrinx.iconset" -o "$out/Syrinx.icns"
        ;;
    windows)
        # Windows icons sit closer to the edge than the Mac grid. One PNG per size,
        # packed into an .ico (PNG entries are fine on Windows Vista and later).
        ico="../../windows/SyrinxManager/assets/Syrinx.ico"
        mkdir -p "$(dirname "$ico")"
        sizes=(16 20 24 32 40 48 64 256)
        for s in $sizes; do
            render_mac $((s * 960 / 1024)) $s "$work/$s.png"
        done
        python3 - "$ico" "$work" $sizes <<'PY'
import struct, sys, pathlib
out, work, sizes = sys.argv[1], pathlib.Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
pngs = [(s, (work / f"{s}.png").read_bytes()) for s in sizes]
head = struct.pack("<HHH", 0, 1, len(pngs))
offset, entries, data = 6 + 16 * len(pngs), b"", b""
for s, png in pngs:
    entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset)
    offset += len(png); data += png
pathlib.Path(out).write_bytes(head + entries + data)
PY
        echo "Wrote windows/SyrinxManager/assets/Syrinx.ico"
        ;;
    *)
        echo "Usage: $0 mac <output dir> | windows" >&2
        exit 1
        ;;
esac
