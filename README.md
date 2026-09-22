# LunarSim

Bağımsız, yeniden kullanılabilir Ay arazi/sensör simülasyon kütüphanesi.
`lunarsim/core` Isaac'e bağımlı değildir (sadece numpy/scipy/rasterio/skyfield);
`lunarsim/adapters/isaac` bunu bir Isaac Sim USD sahnesine bağlar.

Detaylı tasarım kararları için `LUNARSIM_PLAN.md`'ye bakın (Downloads'ta).

## Kurulum

**1) Core (Isaac gerekmez, arazi/ışık/LiDAR/toz/RL üretimi ve testleri için yeterli):**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -v   # 59 test, hepsi geçmeli
```

**2) Isaac Sim / Isaac Lab (gerçek fizik/render doğrulaması ve `scripts/isaac*`/`scripts/isaaclab*` için gerekli):**

```bash
./scripts/install_isaac_docker.sh
```

Bu script sırayla: docker + NVIDIA Container Toolkit'i kontrol eder/kurar (gerekirse
`sudo` ister), `nvcr.io/nvidia/isaac-sim:6.0.1`'i çeker (~10 GB), üstüne
`lunarsim-isaacsim:6.0.1`'i (core Python bağımlılıkları eklenmiş) build eder,
ardından opsiyonel olarak `lunarsim-isaaclab:6.0.1`'i (gerçek Isaac Lab checkout +
torch/cu128, ~10-15 GB daha, kamera render ve çoklu-env hız testleri için gerekli)
build eder. Sadece core+Isaac Sim istiyorsan `LUNARSIM_SKIP_ISAACLAB=1` ile
Isaac Lab aşamasını atlayabilirsin.

**Farklı bir cihazda / diskte kurmak istiyorsan** (ör. ana disk yerine
`/media/DISK02` gibi harici bir diske indirmek), iki ayrı yer var:

1. **Bizim runtime cache'imiz** (shader/GL cache, pip cache, Isaac'in kendi
   ayarları — görece küçük, birkaç GB'a kadar çıkabilir): `LUNARSIM_ISAAC_CACHE`
   env değişkeniyle kontrol edilir, varsayılan `~/docker/isaac-sim`.
   ```bash
   export LUNARSIM_ISAAC_CACHE=/media/DISK02/lunarsim-isaac-cache
   ./scripts/install_isaac_docker.sh
   # sonraki her çalıştırmada da aynı env değişkenini ver (veya ~/.bashrc'ye ekle)
   ```
2. **Asıl büyük veri — docker image'ların kendisi** (~20-25 GB toplam:
   base Isaac Sim image + torch/Isaac Lab layer'ı): bunlar `LUNARSIM_ISAAC_CACHE`'den
   BAĞIMSIZ olarak Docker'ın kendi veri dizinine (`/var/lib/docker`, varsayılan
   olarak her zaman ana diskte) yazılır. Bunları da DISK02'ye almak için Docker'ın
   `data-root`'unu taşımak gerekiyor (tek seferlik, sistem geneli bir ayar):
   ```bash
   sudo systemctl stop docker
   sudo mkdir -p /media/DISK02/docker-data
   # varsa mevcut docker verisini taşı (opsiyonel, temiz kurulumda gerek yok):
   sudo rsync -axP /var/lib/docker/ /media/DISK02/docker-data/
   sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
   { "data-root": "/media/DISK02/docker-data" }
   EOF
   sudo systemctl start docker
   docker info | grep "Docker Root Dir"   # /media/DISK02/docker-data göstermeli
   ```
   Bundan sonra `./scripts/install_isaac_docker.sh` normal şekilde çalışır, tüm
   image'lar DISK02'ye iner. `/media/DISK02` yolunu kendi disk adına göre değiştir.

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

Point cloud'u dosyaya yazmak için `export_point_cloud` (`.ply`/`.npy`/`.csv`):

```python
from lunarsim.core.metadata.lidar import LidarScanPattern, raycast_lidar, export_point_cloud

pattern = LidarScanPattern.spinning(n_channels=32, vertical_fov_deg=(-70, -5), horizontal_res_deg=1.0)
pc = raycast_lidar(tile.height, tile.res_m, sensor_pos, pattern.ray_directions(), max_range_m=60.0)
export_point_cloud(pc, "out/scan.ply")   # CloudCompare/MeshLab/Blender'da direkt açılır
```

Gerçek RTX LiDAR çıktısı için aynı formatlarda `adapters.isaac.sensors.export_rtx_point_cloud`
kullanılabilir (bu ortamda RTX LiDAR çöktüğü için doğrulanamadı, bkz. `lunarsim/adapters/isaac/README.md`).

## Alan bazlı detay kontrolü (Gaussian ROI)

Bir bölgenin detay seviyesini diğerlerine göre artırmak (plan bölüm 5) —
`terrain.roi.regions` ile her bölgeye kendi merkezini (dünya metre cinsinden),
yarıçapını (`sigma_m`) ve gücünü (`weight`, 1.0=tam ince detay) ver:

```yaml
terrain:
  mode: blend
  roi:
    regions:
      - {x_m: 90,  y_m: 60,  sigma_m: 25, weight: 1.0}   # dar, tam detay
      - {x_m: -80, y_m: -50, sigma_m: 60, weight: 0.6}   # geniş, orta detay
```

Kalan her yer coarse kaynağa (gerçek DEM ya da procedural) düşer. Eski
`roi.centers: random_16` / tek `sigma_m` biçimi de hâlâ çalışıyor (tüm ROI'ler
aynı yarıçap/güç).

## Arazi gerçekçilik kontrolü

```python
from lunarsim.core.terrain import check_tile
report = check_tile(tile)  # NaN, aşırı eğim, imkansız krater derinliği vb. yakalar
```

## Toz (hareket eden)

Ay'da atmosfer yok → sürtünme (drag) yok → fırlatılan toz tam bir balistik
parabol izler (gerçek Apollo iniş tozu gözlemlerinin nedeni tam olarak bu:
hava direnci olmadığı için toz çok uzun ve alçak yaylar çizer).
`lunarsim/core/dust/plume.py` bunu tam kinematikle (uydurma bir parçacık
efekti değil) hesaplar; `adapters/isaac/dust.py` bunu gerçek Isaac Sim'de
zaman-örneklenmiş `PointInstancer` pozisyonları olarak authoring eder.
Gerçek Isaac Sim'de doğrulandı: 410 parçacıklı bir patlama 265 zaman
örneği üretti, düşük açılı ~12.6 m/s parçacıklar 89 m'ye kadar gitti (1.62
m/s² yerçekimi altında beklenen menzil).

```python
from lunarsim.adapters.isaac.dust import dust_event_from_disturbance, spawn_dust_burst

event = dust_event_from_disturbance(origin_m=np.array([0, 0, 0.1]), intensity=0.8)
spawn_dust_burst(stage, "/World/Dust", event, prototype_path, tile.height, tile.res_m)
```

## RL (herhangi bir model/algoritma takılabilir)

`lunarsim/rl/analytic_lander_env.py` — standart bir `gymnasium.Env`
(2 eksenli gimbal'lı motor, gerçek TVC dinamiği: gimbal önce gövdeyi
döndürür, gövde eğimi de itki yönünü değiştirir). Isaac'e bağımlı değil
(hızlı, CPU-only iterasyon için), aynı gerçek `core.terrain` heightmap'i
kullanır. Gerçek bir Stable-Baselines3 PPO ile uçtan uca test edildi:
4 paralel env, **~3200 step/s** (CPU).

```python
from lunarsim.rl import AnalyticLanderEnv, LanderParams
from stable_baselines3 import PPO

env = AnalyticLanderEnv(tile, params=LanderParams())
model = PPO("MlpPolicy", env).learn(total_timesteps=100_000)
```

Kendi reward fonksiyonunu geçirebilirsin (`reward_fn=...`); varsayılan
`default_reward_fn` sadece bir başlangıç noktası. Isaac üzerinde tam
fiziksel simülasyon (gerçek temas dinamiği, toz, kamera/LiDAR) isteyen bir
RL görevi için `adapters/isaac/` üzerine bir Isaac Lab `DirectRLEnv` inşa
edilebilir (bkz. `lunarsim/adapters/isaac/README.md`daki Isaac Lab notları).

**Ölçekli eğitim mimarisi (LunarRocket'ın eski koduna bakılarak):** Bulk RL
eğitimi için **her env'e ayrı gerçek DEM/USD terrain mesh'i basmaya gerek
yok**. LunarRocket'ın kendi kodunda `legacy_terrain_usd_max_envs = 16` diye
bir sabit var — gerçek DEM mesh'i sadece 16 env'e kadar spawn ediliyordu,
512 env'lik asıl eğitimde ise `app/terrain_pool.py`'daki **tamamen
parametrik** (sinüs yükseklik fonksiyonu + parametrik krater/kaya, GPU'da
torch tensor, hiç fizik mesh'i yok) bir "terrain pool" kullanılıyordu; gerçek
mesh+temas sadece pretraining sonrası küçük ölçekli fine-tuning/eval için
ayrılmıştı. `AnalyticLanderEnv` zaten bu mimariyle birebir aynı prensipte
(height-lookup, mesh yok) — bu yüzden 256+ paralel env'e sorunsuz ölçekleniyor
(yukarıdaki hız tablosuna bak). Isaac/PhysX tarafında da artık ölçtük: paylaşılan
**tek** gerçek terrain + çoklu rigid body de mükemmel ölçekleniyor (256 env'de
289k step/s). Öneri: bulk eğitim için analitik backend ya da paylaşılan-terrain
Isaac yaklaşımı; gerçek-temas doğrulaması/fine-tuning için küçük (≤16) env
sayısıyla benzersiz gerçek DEM mesh'i — LunarRocket'ın kanıtlanmış deseni bu.

## Test

```bash
.venv/bin/python -m pytest tests/ -v
```

## Isaac Sim doğrulaması

Sıfırdan kurulum: `./scripts/install_isaac_docker.sh` (Docker + NVIDIA Container
Toolkit + `lunarsim-isaacsim:6.0.1` + opsiyonel `lunarsim-isaaclab:6.0.1` build eder).
Bu makinede RTX 5060 Laptop GPU'ya karşı çalıştırıldı ve düzeltildi:

```bash
scripts/run_all_isaac_validations.sh   # hepsi tek komutta: yapısal + fizik + LiDAR + toz + ışık + hız
# veya tek tek:
scripts/run_isaac_smoke_test.sh scripts/isaac_smoke_test.py          # yapısal: her authoring fonksiyonu
scripts/run_isaac_smoke_test.sh scripts/isaac_validation_suite.py    # davranışsal: LiDAR/fizik hızı
scripts/run_isaac_smoke_test.sh scripts/isaac_test_rock_collision.py # kaya collision (gerçek top düşürme)
scripts/run_isaac_smoke_test.sh scripts/isaac_test_dust.py           # toz parçacık yörüngeleri
scripts/run_isaac_smoke_test.sh scripts/isaac_test_sun_rotation.py   # güneş ışığı yön matematiği
# kamera (Isaac Lab imajı gerekir, ./scripts/install_isaac_docker.sh ile build edilir):
scripts/run_isaaclab_camera_test.sh
# çoklu-env hız testi (paylaşılan gerçek terrain, Isaac Lab InteractiveScene):
scripts/run_isaaclab_speed_test.sh 16 64 256
# analitik backend hız testi (Isaac gerekmez):
.venv/bin/python scripts/benchmark_analytic_env_speed.py
```

**Doğrulanan sonuçlar:**
- Heightfield collision + ince render mesh + physics/visual materyal + güneş ışığı + kaya instancing + kamera/LiDAR prim'leri: 13/13 authoring adımı gerçek sahnede hatasız çalışıyor.
- LiDAR ground truth (`raycast_lidar`) gerçek PhysX raycast ile **0.37 mm** ortalama sapmayla eşleşiyor.
- Fizik step hızı: tek env, sadece fizik (render kapalı), **~1900-2100 step/s** (RTX 5060, 8GB VRAM).
- **Çoklu-env hız testi** (paylaşılan tek gerçek terrain + N klonlanmış rigid body, Isaac Lab `InteractiveScene`): 16 env → 23.098 env-step/s, 64 env → 53.740 env-step/s, **256 env → 288.832 env-step/s**. Analitik (Isaac'siz) `AnalyticLanderEnv` backend'i subprocess paralelliğiyle 16 env'de 39.317, 64 env'de 47.105 env-step/s'ye ulaşıyor.
- **Kaya collision fiziksel olarak doğrulandı**: 2m'lik gerçek bir kayanın üstüne 10m'den bırakılan top, tam beklenen temas yüksekliğinde (z=2.200m) durdu.
- **Toz parçacıkları gerçekten hareket ediyor**: 410 parçacıklı patlama, gerçek balistik yörüngeyle 265 zaman örneği, 89m'ye varan menzil (drag yok, tam vakum kinematiği).
- **RL arayüzü uçtan uca çalışıyor**: gerçek bir SB3 PPO, `AnalyticLanderEnv` üzerinde 4 paralel env ile ~3200 step/s eğitim yapıyor.
- Bu süreçte gerçek hatalar bulundu ve düzeltildi: kaya prim'leri açık `Xform` tipiyle authoring edilirse referans edilen asset'in gerçek tipini gölgeliyordu; `PointInstancer.CreateAttribute` diye bir şey yok (`GetPrim().CreateAttribute` gerekiyor); çift physics scene LiDAR sonuçlarını bozuyordu.
- **Kamera render çözüldü**: bare `isaacsim.core.api.World` ile RGB frame hiç ilerlemiyordu; gerçek Isaac Lab `SimulationContext` + `isaaclab.sensors.camera.Camera` ile (`docker/Dockerfile.isaaclab`, `scripts/isaaclab_test_camera.py`) çözüldü. Gerçek 240×320 RGB görüntü üretildi ve güneş açısıyla sinyal **fiziksel olarak doğru şekilde** arttı (12-bit DN ortalaması: 2°→516, 15°→1003, 45°→1735, 80°→1964) — düşük güneş açısında sensöre gerçekten az ışık ulaşıyor (güney kutbu gibi zor aydınlatma koşullarının gerçek fiziksel karşılığı, render hatası değil).

## Depo yapısı

```
lunarsim/
  core/              # Isaac'e bağımlı DEĞİL
    terrain/         # fBm, DEM ingestion, krater, kaya, blend, eğrilik, gerçekçilik kontrolü
    lighting/         # güneş (ephemeris), horizon map, regolith BRDF, kamera gürültüsü
    metadata/         # LiDAR analitik raycast + ground truth kıyası
    quality/         # profil yükleyici
    dust/            # balistik toz kinematiği (Isaac'e bağımlı değil)
  adapters/isaac/    # USD/heightfield export, materyal, ışık, sensör, toz (gerçek Isaac Sim'de doğrulandı)
  rl/                # gymnasium ortamı (analitik, Isaac'e bağımlı değil, SB3 ile test edildi)
configs/             # YAML preset'ler (mare, highland, güney kutbu)
docker/              # sıfırdan Isaac Sim / Isaac Lab image build dosyaları
tests/               # seed tekrarlanabilirliği, istatistik testleri, DEM/LiDAR/ışık/toz/RL testleri
benchmarks/          # profil başına üretim süresi ölçümü
```

## Durum

Uygulanan aşamalar (plan bölüm 11): 1–4 (arazi + blend + testler + kalite
profilleri), 6 (güneş/horizon), 7 (regolith BRDF), 5/8/9 (Isaac adapter —
heightfield/materyal/ışık/kaya authoring + fizik + LiDAR **gerçek Isaac Sim'e
karşı doğrulandı**; kamera authoring doğru ama render pipeline'ı çalışmıyor,
bkz. yukarısı), 10 (preset'ler).

Kalan/eksik:
- **RTX LiDAR denendi, çalışmadı**: gerçek API'ye karşı implemente edildi
  (`isaacsim.sensors.experimental.rtx`, NVIDIA'nın kendi test suite'iyle birebir
  aynı kalıp, gerçek Ouster OS0 donanım profili) ama iki bring-up yolu da (bare
  `World` → hep "Invalid magic number"/geçersiz veri; Isaac Lab `SimulationContext`
  → segfault) başarısız oldu — bu ortamdaki Isaac Sim 6.0.1'in "experimental"
  RTX LiDAR extension'ında gerçek bir kararsızlık. Analitik `raycast_lidar`
  (CPU, gerçek PhysX'e karşı 0.37mm doğrulukla test edildi) kullanılmalı.
  Detay: `lunarsim/adapters/isaac/README.md`.
- Hapke BRDF için özel MDL shader yazılmadı (sadece UsdPreviewSurface yaklaşığı var).
- Isaac'te bazı açılarda görülen küçük speküler parlama artefaktı ayarlanmadı
  (bkz. `scripts/isaaclab_orbit_demo.py` commit notu) — kozmetik, fonksiyonel değil.
- Toz/plume görsel efekti var (`core/dust`, Isaac'te doğrulandı) ama plume/wheel
  olaylarına otomatik bağlanmadı (elle `spawn_dust_burst` çağırman gerekiyor);
  orbital katman, ikinci adapter (MuJoCo/Gazebo), gerçek Isaac Lab `DirectRLEnv` entegrasyonu yok.
- Gerçek DEM dosyaları bu ortama indirilmedi; testler sentetik ama gerçekçi
  georeferanslı bir GeoTIFF fixture'ı ile `dem.py`'ı doğruluyor
  (`tests/conftest.py::synthetic_lunar_dem`).

## Script indeksi (hepsi bunlar, başka bir cihazda test için)

| Script | Ne yapar | Isaac gerekir mi |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -v` | Tüm birim testleri (59 test) | Hayır |
| `benchmarks/benchmark_terrain.py` | Profil başına arazi üretim hızı | Hayır |
| `scripts/train_ppo_analytic.py` | SB3 PPO ile analitik lander eğitimi | Hayır |
| `scripts/benchmark_analytic_env_speed.py` | Analitik RL env hız testi (çoklu-env) | Hayır |
| `scripts/install_isaac_docker.sh` | Docker + Isaac Sim/Lab image'larını sıfırdan kurar | — (kurulum) |
| `scripts/run_isaac_smoke_test.sh <script.py>` | Herhangi bir `isaac_*.py` script'ini `lunarsim-isaacsim` imajıyla çalıştırır | Evet (Isaac Sim) |
| `scripts/run_all_isaac_validations.sh` | Yapısal+fizik+LiDAR+toz+ışık doğrulamalarının hepsi, tek komut | Evet (Isaac Sim) |
| `scripts/isaac_smoke_test.py` | Her authoring fonksiyonunun gerçek sahnede hatasız çalıştığını doğrular | Evet (Isaac Sim) |
| `scripts/isaac_validation_suite.py` | LiDAR-vs-PhysX + fizik hızı + kamera denemesi | Evet (Isaac Sim) |
| `scripts/isaac_test_rock_collision.py` | Gerçek top düşürme ile kaya collision doğrulaması | Evet (Isaac Sim) |
| `scripts/isaac_test_dust.py` | Toz parçacık yörüngelerinin authoring'i | Evet (Isaac Sim) |
| `scripts/isaac_test_sun_rotation.py` | Güneş ışığı yön matematiği (render gerekmez) | Evet (Isaac Sim) |
| `scripts/isaac_test_rtx_lidar.py` | RTX LiDAR (bare World, deneysel) | Evet (Isaac Sim) |
| `scripts/run_isaaclab_camera_test.sh` | Gerçek RGB kamera render doğrulaması (güneş açısı ↔ parlaklık) | Evet (Isaac **Lab**) |
| `scripts/run_isaaclab_speed_test.sh [n ...]` | Çoklu-env fizik hız testi (paylaşılan gerçek terrain) | Evet (Isaac **Lab**) |
| `scripts/run_isaaclab_orbit_demo.sh` | Kamera yörünge demosu → mp4 | Evet (Isaac **Lab**) |
| `scripts/run_isaaclab_live_view.sh` | **Canlı/interaktif** Isaac Sim penceresi (kendi farenle gez) | Evet (Isaac **Lab**), gerçek ekran/DISPLAY gerekir |
| `scripts/isaaclab_test_rtx_lidar.py` | RTX LiDAR (Isaac Lab `SimulationContext` üzerinden) | Evet (Isaac **Lab**) |

`run_isaac_smoke_test.sh` ile başlayanlar hafif `lunarsim-isaacsim` imajını (sadece Isaac
Sim), `run_isaaclab_*` / `isaaclab_*` ile başlayanlar ağır `lunarsim-isaaclab` imajını
(Isaac Lab + torch, kamera piksel yakalama ve `InteractiveScene` gerektiren her şey için
zorunlu — bkz. "RGB kamera render'ı çözüldü" notu) kullanır.
