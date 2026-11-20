#!/bin/bash
# Builds Syrinx.app: the Mac app (this Swift package) with syrinx-server beside it in
# Contents/MacOS, where the app looks for it first. Signed for this Mac only (ad hoc).
#   ./build-app.sh            build/Syrinx.app
#   ./build-app.sh --open     and open it
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
APP="build/Syrinx.app"

CARGO="$(command -v cargo || true)"
[[ -z "$CARGO" && -x /opt/homebrew/opt/rustup/bin/cargo ]] && CARGO=/opt/homebrew/opt/rustup/bin/cargo
[[ -z "$CARGO" && -x "$HOME/.cargo/bin/cargo" ]] && CARGO="$HOME/.cargo/bin/cargo"
if [[ -z "$CARGO" ]]; then
    echo "No cargo: install Rust (brew install rustup) to build syrinx-server." >&2
    exit 1
fi
echo "Building syrinx-server..."
PATH="$(dirname "$CARGO"):$PATH" "$CARGO" build --release -p syrinx-server --manifest-path "$ROOT/Cargo.toml"

echo "Building the app..."
swift build -c release --product SyrinxManager
BIN="$(swift build -c release --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/SyrinxManager" "$APP/Contents/MacOS/Syrinx"
cp "$ROOT/target/release/syrinx-server" "$APP/Contents/MacOS/syrinx-server"
echo "Generating the app icon..."
./make-icons.sh mac "$APP/Contents/Resources"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/crates/syrinx-server/Cargo.toml" | head -1)"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Syrinx</string>
    <key>CFBundleDisplayName</key><string>Syrinx</string>
    <key>CFBundleIdentifier</key><string>com.invursestudio.syrinx</string>
    <key>CFBundleExecutable</key><string>Syrinx</string>
    <key>CFBundleIconFile</key><string>Syrinx</string>
    <key>CFBundleIconName</key><string>Syrinx</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION:-0.1.0}</string>
    <key>CFBundleVersion</key><string>${VERSION:-0.1.0}</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <!-- The statistics page is served by syrinx-server on this Mac, over plain HTTP. -->
    <key>NSAppTransportSecurity</key>
    <dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP/Contents/MacOS/syrinx-server"
codesign --force --sign - "$APP"
echo "Built $(pwd)/$APP"
[[ "${1:-}" == "--open" ]] && open "$APP"
exit 0
