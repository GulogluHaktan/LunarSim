# LunarSim — handover (2026-10-14)

Eski handover `handover_old_2026-10-13.md` olarak saklandı (içindeki Q-gürültüsü kök nedeni ve bazı başlık sayıları yanlış — bkz. memory `lunarsim-handover-errors`). Bu dosya yalnızca bu oturumda **ölçülen** şeyleri içerir.

## 0. Hedef (kullanıcı, /goal)
MPC, ES ve PPO **en az %50 iniş başarısı** almalı. Kullanıcı kararı: **iki araziyi de raporla** — Apollo 16 (zor) ve Apollo 11 (orta); %50 hedefi Apollo 11 için, Apollo 16 için bulunan en iyi sonuç. Teslim: videolar (inişin tamamı, sonda ışınlanma yok) + Türkçe PDF (yöntemler, akış şemaları, grafikler, sonuçlar, kısıtlar). Mobilde indirilebilir; tek dosya ≤ 30 MiB (bölerek gönder). Video sıkıştırma istemiyor. Kullanıcı internet kotasına dikkat ediyor (günlük 8 GB izin verdi, daha fazla indirme için sor).

## 1. Ortam / komutlar
- Python: `/home/haktan/isaac-env/bin/python` (Isaac Sim 6.1; rasterio yok → GeoTIFF'i PIL ile okuyoruz).
- Her koşuda: `export PYTHONPATH=$PWD LUNARSIM_LOLA_IMG=<TIF> LUNARSIM_TILE_POOL=<pool.pkl>`.
- Veri: `~/lunarsim_data/lroc/NAC_DTM_APOLLO16.TIF` ve `..._APOLLO11.TIF` (2 m/px LROC NAC DTM, `.f32.npy` önbellekli).
- Tile havuzları: `out/a16/pool_{train,val}.pkl` (train = batı yarı lon 15.176–15.508, val = doğu yarı 15.574–15.904), `out/a11/pool_{train,val}.pkl` (val=24 tile; `pool_val48.pkl` yapılıyorsa 48). Grid 384 (4.4 m), orbit_descent 1680 m.
- Kullanılan stage: `orbit_descent` (200 m, 10–30 m/s yanal, 60 s bütçe, 20 Hz). İniş kriteri: |vz|≤1.0, vxy≤1.2, tilt≤15°, |w|≤0.5, ayak-pedi yükseklik farkı ≤0.16 m (pad yarıçapı 4.7 m, 4 nokta).
- `pkill -f` kendi shell'ini öldürür (komut metni pattern'i içerir) → `pkill -f "pat[t]ern"` biçimi kullan. Zsh'te `$args` kelime bölmez → argümanları ayrı yaz.
- Hiçbir şey commit edilmedi (git status uzun). Commit isteyince yap.

## 2. Arazi gerçeği (en önemli bulgu)
Güvenli-hücre oranı (footpad farkı ≤0.12 m) val havuzu ortalaması: **Apollo 16 %0,6** (11/24 tile'da <%0,2), **Apollo 11 %8,8**. DTM'de çöp yükseklik (1e22 m'ye kadar) vardı: PhysX'i çökertiyordu → `lola.py`'da validity ±1e4 m + medyan-spike filtresi + site için ≥%99,5 geçerli piksel.
- Hazard haritası: `lunarsim/core/terrain/hazard.py` (`diff`=footpad spread, `dist`=en yakın güvenli hücreye EDT mesafesi; `LUNARSIM_HAZ_DIFFMAX` eşiği geçersiz kılar).
- Kör (araziyi görmeyen) politikaların tavanı ≈ güvenli hücre oranı.

## 3. Ölçülen sonuçlar (Isaac, aksi belirtilmedikçe val havuzu, 96 bölüm)
**Apollo 16 (zor):**
| Yöntem | İniş |
|---|---|
| ZEM/ZEV kontrolcü | 0/96 |
| Kör ES / kör MPC / kör PPO | 1 / 2 / 1 /96 |
| MPC + hazard terimi (eski ağırlıklar, düz-zemin rollout) | 5/96 (%5,2) |
| ES-haz (27-d doğrusal, kötü havuzda eğitilmişti) | 0/96 |
| PPO-haz (hazard özellikli guided-gain) | 1/96 |
| Hedef-noktalı ZEM/ZEV kuralı (öğrenme yok, "aim") | 5/96 (%5,2) |
| ES (aim uzayında, 13 sabit sayı) | 5/96 (%5,2) |
| **MPC v3 (arazi-farkında rollout + yeniden dengelenmiş ağırlık, `out/a16/mpc_v3.npy`)** | **17/96 (%17,7)** |
| MPC v3b / v3c (w_time 0,73 / 1,2) | 15/96 / 13/96 |
- PPO-aim eğitimi (`out/a16/ppo_rule`, 90k karar): seçim bloğu 12,5/8,3/2,1/6,2/4,2% — taban 4,2%, gürültü içinde; doğrulaması **yapılmadı** (`best_on_selection_block.zip`).
- Damıtma (MPC→MLP, DAgger): öğrenci %0–8 (öğretmen %48–58 analitik). Çalışmıyor.
- ES (linear/MLP) analitikte 150–400 nesil: seçim fitness'ı iyileşiyor, iniş kural tabanını geçmiyor.

**Apollo 11 (orta):** analitik (48 bölüm) MPC v3 **32/48 = %67**, aim-kuralı 11/48, kontrolcü 4/48. **Isaac MPC v3 doğrulaması çalışıyor** (`out/a11/mpc_v3.log`, ara: 20/36). ES/PPO henüz A11'de denenmedi.

## 4. Bu oturumda bulunan kök nedenler (MPC'yi 5% → 18% yapan)
1. **MPC rollout'u zemini düz varsayıyordu.** Yüksek arazide yanal hızla zemin yükseliyor → touchdown zamanı yanlış, sert iniş. Düzeltme: `MPPIMPC.set_hazard(..., height_maps)` + `_hgt_at` ile rollout'ta `alt = alt0 + dz − (h(px,py) − h0)`; `LUNARSIM_MPC_TERRAIN=0` kapatır.
2. **Tehlike maliyeti iniş yumuşaklığından ~300× büyüktü** (w_haz log 2,58 vs w_td log 0,11) → MPC güvenli hücreye sert iniyordu (vxy sınırın 2,5 katı). `w_td=0.3, w_haz=2.2, w_time=0` analitikte 5/36 → 21/36.
3. `w_time`>0 + `k_ref` yükseltmek analitikte crash'e çevirir (k_ref yalnız yüksek irtifada geçerli olsun diye `_target`'te `LUNARSIM_MPC_KBLEND` karışımı eklendi; kazanç yok/nötr).
4. Seçim bloğu (12 bölüm) her nesilde sabitti → eski "SEL %67–83" **aşırı uyum**; gerçek analitik val ≈ %14 (eski ağırlık) / %37 (v3, taze tohum).
5. Isaac ≠ analitik: v3 analitik ~%37–58, Isaac %18 (Apollo 16). 100 s bütçeyle Isaac'ta crash'lerin 28/29'u `leg_diff` (yani site hassasiyeti), süre aşımı değil.
6. Kontrolcü zaten yumuşak iniyor (vz 0,8, vxy 0,6, tilt 7°); 0/N sadece **site** yüzünden. Yanal kanalı yalnız hız sönümlemedir → ES/PPO "nereye ineceğini" etkileyemiyordu. Çözüm denemesi: kontrolcüye `v_ref`, `aim_pos` (tam ZEM/ZEV hedef noktası), `pos_fade_alt_m`, `final_pos_kp`, `terrain_ahead` (varsayılan kapalı, eski davranış korunur).

## 5. Kod değişiklikleri (hepsi commit edilmemiş)
- `lunarsim/core/terrain/{lola,hazard,generate,craters}.py` — NAC/LOLA DEM, hazard haritası, krater taper.
- `lunarsim/evo/mpc.py` — `w_haz` (11. parametre), hazard + yükseklik rollout'u, env: `LUNARSIM_HAZ_SCALE`, `LUNARSIM_MPC_TERRAIN`, `LUNARSIM_MPC_KBLEND`.
- `lunarsim/evo/hazfeat.py` — 7 radar özelliği, `safe_dir`, `aim_point`.
- `lunarsim/control/zemzev_controller.py` — aim alanları (varsayılanda eski davranış).
- `lunarsim/rl/guided_gain_env.py`, `guided_gain_vec_env.py` — `aim_scale`/`LUNARSIM_AIM_SCALE`, `hazard_obs`/`LUNARSIM_HAZ_OBS`, `LUNARSIM_HAZ_LAMBDA_PPO` şekillendirme, truncated `terminal_observation` genişliği düzeltmesi.
- `lunarsim/rl/curriculum.py` — havuz örnekleyici, `LUNARSIM_TMAX` (tanı amaçlı bütçe geçersiz kılma).
- Scriptler: `build_tile_pool.py`, `add_hazard_to_pool.py`, `evo_analytic_train_haz.py`, `evo_guided_aim_train.py` (ES, aim uzayı; `--bias-only 1`), `mpc_tune.py`, `mpc_analytic_eval.py` (`LUNARSIM_MPC_HANDOFF`, `LUNARSIM_TMAX`), `mpc_isaac_eval.py`, `mpc_record_isaac.py`, `mpc_distill_dagger.py`, `aim_rule_probe.py`, `validate_guided_gain_isaac.py` (+ `.npz` ES desteği), `record_guided_gain_isaac.py` (+ `.npz` ES).

## 6. Yapılacaklar (sırayla)
1. `out/a11/mpc_v3.json` bitince A11 Isaac MPC sonucunu not et (hedef ≥%50).
2. **A11'de ES ve PPO'yu başarılı yap (≥%50).** Denenmemiş fikirler: (a) A11'de aim-uzayında ES/PPO (kural tabanı analitikte %23, süre aşımı çok), (b) A11 havuzunda DAgger damıtma (hedef daha toleranslı, öğrenci A16'dan iyi olabilir), (c) "MPC ağırlıklarını modüle eden PPO" / ES-ayarlı MPC — bu **kontrolcüsüz politika değildir**, raporda açıkça etiketle. Dürüst raporla: hangisi gerçekten öğrenilmiş politika.
3. Isaac'ta paired doğrulama (8 dalga × 12 = 96; McNemar, Wilson CI). `pool_val48` kullanılacaksa dalga başına farklı tile gelir; tohumları 90000..97000.
4. Videolar: dalga 96000, env 1 (A16'da kural+ES+MPC üçü de indi); A11 için aynı yöntemle uygun env'i seç. Pipeline: `evo_extend_recording.py` → `isaaclab_replay_capture.py` → `synth_lidar_scans.py` → `compose_capture_video.py` → `evo_compose_dashboard_video.py` (bkz. `out/crater2/run.sh`).
5. PDF: `report/build_krater_rapor.py` genişlet — arazi (LOLA/NAC, güvenli hücre oranı), neden kör politikalar tavanda, hazard haritası, MPC arazi-farkında rollout + ağırlık dengesi, aim kontrolcüsü, ES/PPO sonuçları ve **negatif sonuçlar** (damıtma, ES-haz, PPO-haz), Isaac–analitik farkı, sınırlar (hazard haritası önsel DEM bilgisi; train/val coğrafi ayrık; n=96 geniş CI; MPC dinamiği biliyor).
6. `lunarsim_krater_paket_3.zip` (PPO videosu) hiç teslim edilmedi — hatırlat.

## 7. Dikkat
- Analitik sayılar iyimser (Isaac'a göre 2×); karar sayısı Isaac'tan çıkmalı.
- Tarama ile seçilen ağırlıkların analitik skorları seçim yanlılığı taşır; taze tohumla ölç.
- `out/` daha önce silinmişti (memory `lunarsim-out-wiped`): önemli sonuçları (`out/a16/mpc_v3.npy`, `out/a16/final/*.json`, `out/a11/*`) repo dışına da yedekle.
