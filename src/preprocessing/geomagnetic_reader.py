"""
geomagnetic_reader.py
=====================
Refactored dari: awal/read_mdata.py

Pembaca data biner geomagnetik format FRG604RC (BMKG).
Mendukung format .STN (uncompressed) dan .gz (gzip-compressed).

Perubahan dari versi awal:
  - Diubah menjadi kelas GeomagneticReader (OOP, reusable)
  - Ditambahkan dataclass GeomagneticData untuk output terstruktur
  - Ditambahkan validasi coverage dan statistik
  - Ditambahkan dukungan multi-stasiun via load_multistation()
  - Backward compatible: fungsi read_604rcsv_new_python() tetap tersedia
"""

import os
import gzip
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA FORMAT BINER FRG604RC
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_HEADER_SIZE = 30       # bytes per block header
RECORD_SIZE       = 17       # bytes per 1-second record
RECORDS_PER_BLOCK = 600      # records per block
BLOCK_SIZE        = BLOCK_HEADER_SIZE + RECORD_SIZE * RECORDS_PER_BLOCK  # 10230
SECONDS_PER_DAY   = 86400

# Threshold validitas fisik
THRESHOLD_HDZ     = 80000.0  # nT — threshold H, D, Z
THRESHOLD_IXY     = 3000.0   # threshold IX, IY
THRESHOLD_TEMP    = 300.0    # °C — threshold suhu
THRESHOLD_VOLTAGE = 24.0     # V — threshold tegang