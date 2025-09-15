#!/bin/bash
# Script to clean all PLOTS directories inside STATIONS/* down to 5 levels

# Base directory (adjust if needed)
BASE_DIR="$HOME/DATAFLOW_v3/STATIONS"

# Check if base dir exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory $BASE_DIR does not exist."
  exit 1
fi

# Calcular tamaño antes (en KB)
before=$(du -sk "$BASE_DIR" | awk '{print $1}')

echo "Cleaning all PLOTS directories under: $BASE_DIR"

# Find all directories named "PLOTS" up to 5 levels deep
find "$BASE_DIR" -type d -name "PLOTS" -maxdepth 5 | while read -r dir; do
  echo "→ Cleaning $dir"
  # Remove all files inside the PLOTS directory but keep the folder itself
  rm -rf "$dir"/*
done

echo "All PLOTS directories cleaned."

after=$(du -sk "$BASE_DIR" | awk '{print $1}')
freed=$((before - after))

# Formateo en GB (10^9) y GiB (2^30)
before_gb=$(awk -v v="$before" 'BEGIN{printf "%.2f", v/1000000}')
after_gb=$(awk -v v="$after"   'BEGIN{printf "%.2f", v/1000000}')
freed_gb=$(awk -v v="$freed"   'BEGIN{printf "%.2f", v/1000000}')

before_gib=$(awk -v v="$before" 'BEGIN{printf "%.2f", v/1024/1024}')
after_gib=$(awk -v v="$after"   'BEGIN{printf "%.2f", v/1024/1024}')
freed_gib=$(awk -v v="$freed"   'BEGIN{printf "%.2f", v/1024/1024}')

echo "✅ Limpieza completada."
echo "   Antes:  ${before_gb} GB  (${before_gib} GiB)"
echo "   Después: ${after_gb} GB  (${after_gib} GiB)"
echo "   Liberado: ${freed_gb} GB  (${freed_gib} GiB)"
