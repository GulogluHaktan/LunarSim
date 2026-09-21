# LunarSim

Bağımsız, yeniden kullanılabilir Ay arazi/sensör simülasyon kütüphanesi.
`lunarsim/core` Isaac'e bağımlı değildir (sadece numpy/scipy/rasterio/skyfield);
`lunarsim/adapters/isaac` bunu bir Isaac Sim USD sahnesine bağlar.

Detaylı tasarım kararları için `LUNARSIM_PLAN.md`'ye bakın (Downloads'ta).

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Gerçek Ay verisi (önemli)

Arazi **uydurma değil, gerçek NASA DEM verisinden** üretilir. Coarse (büyük
ölçek) katman `lunarsim/core/terrain/dem.py` ile gerçek bir LOLA/LRO
rasterından okunur; sadece kraterler/kayalar/yüksek-frekans pürüz gibi
DEM'in çözemediği detaylar procedural olarak eklenir (bu, gerçek DEM
tabanlı simülasyon veri kümelerinde standart bir uygulamadır, sahayı
"uydurmak" değildir).

Desteklenen kaynaklar (rasterio/GDAL açabildiği her GeoTIFF):
- **LOLA GDR mozaikleri** (118 ppd / 512 ppd), PDS Geosciences Node
- **LOLA+Kaguya TC birleşik DEM** (59 m/px), USGS Astropedia
- **LRO NAC DTM'leri** (0.5–5 m/px, tek site), PDS Cartography/Imaging Node

Bir tile'ı gerçek DEM'den üretmek için `configs/example.yaml`'daki gibi
`terrain.coarse_source: dem` + `terrain.dem_path` + `site_lat_deg`/`site_lon_deg`
verin. `coarse_source: procedural` (fBm) **sadece** bölgenin önemsiz olduğu
RL eğitim tile havuzu için (`configs/training_procedural.yaml`) kullanılmalı;
gerçek bir sahanın yerine asla geçirilmemeli.

## Hızlı kullanım

```python
from lunarsim.core.terrain import TerrainConfig, generate_tile, save_tile

cfg = TerrainConfig.from_yaml("configs/example.yaml")
tile = generate_tile(cfg)
save_tile(tile, "out/my_tile")
```

Çıktı: `heightmap.npy` (float32, metre) + `meta.json` (seed, parametreler,
sürüm, mod) + `ground_truth.json` (krater/kaya konum-boyut listeleri). Aynı
seed = birebir aynı çıktı (bkz. `tests/test_terrain.py`).

## Kalite profilleri

```python
from lunarsim.core.quality import load_quality

q = load_quality("configs/presets/south_pole.yaml")
print(q.profile, q.render, q.lidar)
```

`fast | balanced | high | reference` — donanım değişince tek satır (`quality.profile`)
değiştirilir, kod hiçbir yerde donanıma bakmaz (bkz. `lunarsim/core/quality/default_profiles.yaml`).

```bash
.venv/bin/python benchmarks/benchmark_terrain.py --size-m 500 --out benchmarks/benchmarks.csv
```

## Güneş/gölge

```python
from datetime import datetime, timezone
from lunarsim.core.lighting import SunEphemeris, horizon_profile, sun_visible

eph = SunEphemeris()  # ilk çağrıda DE421 kernelini .cache/ altına indirir
pos = eph.sun_position(datetime(2026, 9, 21, 12, tzinfo=timezone.utc), lat_deg=-89.0, lon_deg=0.0)
```

Gerçek JPL ephemeris'i (Skyfield + DE421) ile Ay'ın IAU rotasyon modelini
birleştirerek gerçek tarih/site için güneş elevation/azimuth verir.
`horizon_profile` heightmap'ten bağımsız bir ufuk haritası hesaplayarak
render gölgelerini doğrulamak için kullanılabilir.

## LiDAR ground truth

`lunarsim/core/metadata/lidar.py` heightfield üzerinde analitik ray-casting
yapar (Isaac'in RTX LiDAR'ına bağımlı değildir); `compare_point_clouds` ile
RTX çıktısına karşı menzil hatası ölçülür. Gerçek Isaac Sim 6.0.1'e karşı
doğrulandı: 384 ışından 341'i her iki yöntemde de aynı, ortalama sapma
**0.37 mm**, RMSE 1.58 m (birkaç sınır-açılı ışından kaynaklanıyor) — bkz.
"Isaac Sim doğrulaması" altında.

## Arazi gerçekçilik kontrolü

```python
from lunarsim.core.terrain import check_tile
report = check_tile(tile)  # NaN, aşırı eğim, imkansız krater derinliği vb. yakalar
```

## Test

```bash
.venv/bin/python -m pytest tests/ -v
```

## Isaac Sim doğrulaması

Bu makinede kurulu gerçek bir Isaac Sim 6.0.1 docker imajına (`lunar-rocket-isaaclab:6.0.1`,
RTX 5060 Laptop GPU) karşı çalıştırıldı ve düzeltildi:

```bash
scripts/run_isaac_smoke_test.sh scripts/isaac_smoke_test.py     # yapısal: her authoring fonksiyonu
scripts/run_isaac_smoke_test.sh scripts/isaac_validation_suite.py  # davranışsal: LiDAR/fizik hızı
```

**Doğrulanan sonuçlar:**
- Heightfield collision + ince render mesh + physics/visual materyal + güneş ışığı + kaya instancing + kamera/LiDAR prim'leri: 13/13 authoring adımı gerçek sahnede hatasız çalışıyor.
- LiDAR ground truth (`raycast_lidar`) gerçek PhysX raycast ile **0.37 mm** ortalama sapmayla eşleşiyor.
- Fizik step hızı: tek env, sadece fizik (render kapalı), **~1900-2100 step/s** (RTX 5060, 8GB VRAM).
- Bu süreçte gerçek bir hata bulundu ve düzeltildi: kaya prim'leri açık `Xform` tipiyle authoring edilirse referans edilen asset'in gerçek tipini (görünürlük/collision'ı) gölgeliyordu — artık tip belirtilmeden authoring ediliyor.
- Bilinen eksik: headless docker'da RGB kamera render pipeline'ı `isaacsim.core.api.World` ile hiç ilerlemiyor (frame sayacı 0'da kalıyor); LunarRocket'taki tek çalışan kamera render örneği gerçek bir Isaac Lab `SimulationContext` üzerinden geliyor, bare `World` üzerinden değil. Detay: `lunarsim/adapters/isaac/README.md`.

## Depo yapısı

```
lunarsim/
  core/              # Isaac'e bağımlı DEĞİL
    terrain/         # fBm, DEM ingestion, krater, kaya, blend, eğrilik
    lighting/         # güneş (ephemeris), horizon map, regolith BRDF, kamera gürültüsü
    metadata/         # LiDAR analitik raycast + ground truth kıyası
    quality/         # profil yükleyici
  adapters/isaac/    # USD/heightfield export, materyal, ışık, sensör (yapısal, Isaac gerektirir)
configs/             # YAML preset'ler (mare, highland, güney kutbu)
tests/               # seed tekrarlanabilirliği, istatistik testleri, DEM/LiDAR/ışık testleri
benchmarks/          # profil başına üretim süresi ölçümü
```

## Durum

Uygulanan aşamalar (plan bölüm 11): 1–4 (arazi + blend + testler + kalite
profilleri), 6 (güneş/horizon), 7 (regolith BRDF), 5/8/9 (Isaac adapter —
heightfield/materyal/ışık/kaya authoring + fizik + LiDAR **gerçek Isaac Sim'e
karşı doğrulandı**; kamera authoring doğru ama render pipeline'ı çalışmıyor,
bkz. yukarısı), 10 (preset'ler).

Kalan/eksik:
- Headless docker'da RGB kamera render'ı: authoring doğru, ama gerçek piksel
  üretmiyor — gerçek bir Isaac Lab `SimulationContext` gerektiriyor (bare
  `isaacsim.core.api.World` yetmiyor). Detay: `lunarsim/adapters/isaac/README.md`.
- RTX LiDAR (GPU-hızlandırmalı) hiç implemente edilmedi/doğrulanmadı; analitik
  `raycast_lidar` (CPU, gerçek PhysX'e karşı doğrulandı) onun yerine kullanılabilir.
- Hapke BRDF için özel MDL shader yazılmadı (sadece UsdPreviewSurface yaklaşığı var).
- Çoklu-env (512-1024) fizik hızı ölçülmedi — sadece tek env, ~1900-2100 step/s.
- Toz/plume, orbital katman, ikinci adapter (MuJoCo/Gazebo), RL entegrasyonu.
- Gerçek DEM dosyaları bu ortama indirilmedi; testler sentetik ama gerçekçi
  georeferanslı bir GeoTIFF fixture'ı ile `dem.py`'ı doğruluyor
  (`tests/conftest.py::synthetic_lunar_dem`).
