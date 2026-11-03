#!/usr/bin/env bash
# Download and extract the SimCol3D synthetic colonoscopy dataset from Figshare.
#
# Dataset: https://doi.org/10.5522/04/24077763.v1 (CC BY-NC-SA 4.0)
#
# Usage:
#   scripts/download_simcol.sh [--dest DIR] [--keep-zips] [SCENE ...]
#
# SCENE is one or more of: I II III misc all (default: misc III)
# DEST defaults to $SIMCOL_DIR, or ./data/simcol3d.

set -euo pipefail

BASE_URL="https://ndownloader.figshare.com/files"

# name -> "figshare_file_id md5"
declare -A FILES=(
    [misc]="42270570 4e20ef680dc7c9dca90133fdf29e218b"
    [README]="42248541 3fe466666638a6b19d022f6135ad089f"
    [I]="42248457 f04fa71c7432b7b01518361b210bb358"
    [II]="42270579 da86c7bdd29efa521b34e284bce4b6cd"
    [III]="42270582 fed7558d517ca39364aa75665bda91e9"
)

declare -A ARCHIVES=(
    [misc]="misc.zip"
    [README]="README.txt"
    [I]="SyntheticColon_I.zip"
    [II]="SyntheticColon_II.zip"
    [III]="SyntheticColon_III.zip"
)

dest="${SIMCOL_DIR:-data/simcol3d}"
keep_zips=0
scenes=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) dest="$2"; shift 2 ;;
        --keep-zips) keep_zips=1; shift ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        all) scenes+=(misc I II III); shift ;;
        I|II|III|misc) scenes+=("$1"); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ ${#scenes[@]} -eq 0 ]] && scenes=(misc III)

for cmd in curl md5sum unzip; do
    command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 1; }
done

mkdir -p "$dest"
dest="$(cd "$dest" && pwd)"
echo "Destination: $dest"

download() {
    local name="$1" id md5 file
    read -r id md5 <<< "${FILES[$name]}"
    file="$dest/${ARCHIVES[$name]}"

    if [[ -f "$file" ]] && echo "$md5  $file" | md5sum -c --status; then
        echo "[$name] $file already downloaded"
    else
        echo "[$name] downloading ${ARCHIVES[$name]}"
        curl -L --fail --retry 3 --progress-bar -C - -o "$file" "$BASE_URL/$id"
        echo "$md5  $file" | md5sum -c --quiet || { echo "[$name] MD5 mismatch" >&2; exit 1; }
    fi

    if [[ "$file" == *.zip ]]; then
        echo "[$name] extracting"
        unzip -q -o "$file" -d "$dest"
        [[ $keep_zips -eq 1 ]] || rm -f "$file"
    fi
}

download README
for scene in "${scenes[@]}"; do
    download "$scene"
done

echo "Done."
