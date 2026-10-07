```bash
mkdir -p /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/Kvasir-SEG
cd /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/Kvasir-SEG

wget --no-check-certificate \
  https://datasets.simula.no/downloads/kvasir-seg.zip

ls -lh kvasir-seg.zip
file kvasir-seg.zip

unzip kvasir-seg.zip

find . -maxdepth 3 -type d | sort

```