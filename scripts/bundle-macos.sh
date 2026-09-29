#!/bin/sh
# Build a double-clickable Ember.app from the release binary.
#
#   scripts/bundle-macos.sh            # builds, then writes target/bundle/Ember.app
#   open target/bundle/Ember.app
#
# Why a bundle: run as a bare binary the app has no icon and the menu bar says
# "ember"; as a bundle it gets the icon, the name and a Dock entry.
#
# Models: Ember finds .gguf files by scanning the folder it starts in, and a
# Finder launch starts in "/". The launcher therefore starts in
# $EMBER_MODELS_DIR if set, else the checkout this bundle was built from.
set -eu

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

cargo build --release --bin ember

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
app="target/bundle/Ember.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

cp target/release/ember "$app/Contents/MacOS/ember-bin"
cp assets/macos/Ember.icns "$app/Contents/Resources/Ember.icns"

cat > "$app/Contents/MacOS/ember" <<LAUNCHER
#!/bin/sh
here="\$(cd "\$(dirname "\$0")" && pwd)"
cd "\${EMBER_MODELS_DIR:-$root}" 2>/dev/null || cd "\$HOME"
exec "\$here/ember-bin" gui
LAUNCHER
chmod +x "$app/Contents/MacOS/ember"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Ember</string>
  <key>CFBundleDisplayName</key><string>Ember</string>
  <key>CFBundleIdentifier</key><string>dev.ember.console</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleExecutable</key><string>ember</string>
  <key>CFBundleIconFile</key><string>Ember</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHumanReadableCopyright</key><string>Ember</string>
</dict>
</plist>
PLIST

echo "built $app"
