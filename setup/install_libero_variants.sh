#!/usr/bin/env bash
# install_libero_variants.sh
#
# Clones and pip-installs the three LIBERO variants needed to reproduce our
# eval numbers:
#
#   Standard LIBERO   → https://github.com/Lifelong-Robot-Learning/LIBERO
#   LIBERO-PRO        → https://github.com/Zxy-MLlab/LIBERO-PRO
#   LIBERO-Plus       → https://github.com/sylvestf/LIBERO-plus
#
# All three packages import as `libero`, so they are mutually exclusive within
# a single Python environment. Recommended: one conda env per variant.
#
# Usage
# -----
#   bash setup/install_libero_variants.sh <variant> [--dest <dir>]
#
#   <variant>   one of: libero | libero-pro | libero-plus | all
#   --dest      directory to clone into (default: ../libero-variants)
#
# Examples
# --------
#   # Just install standard LIBERO (only Standard eval works)
#   bash setup/install_libero_variants.sh libero
#
#   # Install LIBERO-PRO (Standard + PRO evals work)
#   bash setup/install_libero_variants.sh libero-pro
#
#   # Install LIBERO-Plus (Standard + Plus evals work)
#   bash setup/install_libero_variants.sh libero-plus
#
#   # Clone all three side-by-side (for switching between them via `pip install -e`)
#   bash setup/install_libero_variants.sh all --dest /path/to/libero-variants
#
# After installation, the eval scripts in this repo (experiments/robot/libero,
# experiments/robot/libero_pro, experiments/robot/libero_plus) will use the
# variant that is currently pip-installed as `libero`.
#
# The Plus eval scripts expect the LIBERO-plus repo root (not the pip-installed
# package) to be reachable via the LIBERO_PLUS_ROOT env var — see REPRODUCE.md.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VARIANT="${1:-}"
DEST="${DEST:-../libero-variants}"

# Parse --dest flag
shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

if [ -z "$VARIANT" ]; then
  echo "Usage: $0 <variant> [--dest <dir>]"
  echo "  <variant> = libero | libero-pro | libero-plus | all"
  exit 1
fi

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
echo "Cloning into $DEST"

clone_one() {
  local name="$1"; local url="$2"
  local target="$DEST/$name"
  if [ -d "$target/.git" ]; then
    echo "[$name] already cloned, skipping"
  else
    echo "[$name] cloning $url"
    git clone "$url" "$target"
  fi
}

install_one() {
  local name="$1"
  local target="$DEST/$name"
  echo "[$name] pip install -e ."
  ( cd "$target" && pip install -e . )
  echo "[$name] installed as 'libero' package"
}

case "$VARIANT" in
  libero)
    clone_one LIBERO         https://github.com/Lifelong-Robot-Learning/LIBERO.git
    install_one LIBERO
    ;;
  libero-pro)
    clone_one LIBERO-PRO     https://github.com/Zxy-MLlab/LIBERO-PRO.git
    install_one LIBERO-PRO
    echo
    echo "LIBERO-PRO installed. To run PRO evals, export:"
    echo "  export LIBERO_PRO_ROOT=\"$DEST/LIBERO-PRO\""
    ;;
  libero-plus)
    clone_one LIBERO-plus    https://github.com/sylvestf/LIBERO-plus.git
    install_one LIBERO-plus
    echo
    echo "LIBERO-plus installed. To run Plus evals, export:"
    echo "  export LIBERO_PLUS_ROOT=\"$DEST/LIBERO-plus\""
    ;;
  all)
    clone_one LIBERO         https://github.com/Lifelong-Robot-Learning/LIBERO.git
    clone_one LIBERO-PRO     https://github.com/Zxy-MLlab/LIBERO-PRO.git
    clone_one LIBERO-plus    https://github.com/sylvestf/LIBERO-plus.git
    echo
    echo "Cloned all three variants side-by-side in $DEST"
    echo "To switch between them (they all install as the 'libero' package):"
    echo "  cd $DEST/LIBERO       && pip install -e .   # Standard only"
    echo "  cd $DEST/LIBERO-PRO   && pip install -e .   # Standard + PRO"
    echo "  cd $DEST/LIBERO-plus  && pip install -e .   # Standard + Plus"
    echo
    echo "Export both env vars so the eval scripts can find the perturbation data:"
    echo "  export LIBERO_PRO_ROOT=\"$DEST/LIBERO-PRO\""
    echo "  export LIBERO_PLUS_ROOT=\"$DEST/LIBERO-plus\""
    ;;
  *)
    echo "unknown variant: $VARIANT (expected libero | libero-pro | libero-plus | all)"
    exit 1
    ;;
esac

echo
echo "Also install VLA-Adapter's LIBERO requirements:"
echo "  pip install -r experiments/robot/libero/libero_requirements.txt"
