
mkdir -p data

for m in {4..12}; do
    mp=$(printf "%02d.csv" $m)
    if [[ -e data/$mp ]]; then
      continue
    fi
    if ! wget https://data.urbansharing.com/oslobysykkel.no/trips/v1/2019/${mp} -O data/${mp}; then
        break
    fi
done
