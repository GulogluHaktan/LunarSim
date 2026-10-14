# LunarSim: Evrimsel öğrenme ile iniş — ajan görev dosyası

Bu dosya, projeyi daha önce görmemiş bir ajanın **mevcut yapıyı bozmadan**
evrimsel yöntemlerle (Evolution Strategies / CMA-ES) iniş öğrenmeyi deneyip
**ölçülmüş** bir sonuç raporlaması için yazıldı. Baştan sona oku, sonra başla.

**Tek cümlelik araştırma sorusu:** RL kullanmadan, sadece uçuşun sonucundan
öğrenen bir evrimsel yöntem (a) kontrolcü kazançlarını ayarlayarak PPO'nun
%89.6'sına ulaşabilir mi, ve (b) kontrolcü olmadan, itkiyi doğrudan komut
ederek sıfırdan inmeyi öğrenebilir mi?

---

## 1. Kesin kurallar (ihlal etme)

1. **Mevcut hiçbir dosyayı değiştirme.** `lunarsim/`, `scripts/`, `tests/`,
   `report/` altındaki var olan dosyalar, `handover.md`, `RESUME.md` dahil.
   Sadece YENİ dosya ekle:
   - `lunarsim/evo/` (yeni paket: `__init__.py`, `policies.py`, `es.py`, `paired.py` ...)
   - `scripts/evo_*.py` (eğitim, doğrulama, teşhis scriptleri)
   - `tests/test_evo_*.py`
   - Mevcut koddan bir fonksiyon lazımsa **import et**; import edilemiyorsa
     (ör. `scripts/validate_guided_gain_isaac.py` modül seviyesinde
     `SimulationApp` başlatıyor) mantığını yeni dosyaya **kopyala** ve
     kaynağını yorumda belirt.
2. **Hiçbir şeyi silme.** `out/` daha önce başka bir ajan tarafından silindi
   ve tüm sonuçlar yeniden üretilmek zorunda kalındı. `out/` altında sadece
   `out/evo/` içine yaz. Önemli her sonucu ayrıca
   `~/lunarsim_artifacts/evo/` içine kopyala.
3. **Git commit atma**, push yapma. Kullanıcı isterse kendisi karar verir.
4. **Tohum blokları ayrılmıştır:**
   - `7000` = PPO'nun checkpoint SEÇİM bloğu. Kullanma.
   - `90000, 91000, …, 105000` (16 dalga) = nihai DOĞRULAMA dalgaları.
     Eğitim veya seçim sırasında **asla** kullanma; sadece en sonda bir kez.
   - Eğitim için `200000+`, ES seçimi (en iyi bireyi seçmek) için `150000–159000` kullan.
5. **Her mekanizma bayrağını nesne üzerinde assert et.** Bu projede dokuz
   bayrak, argümanı yazdıran ama uygulamayan kod yüzünden sessizce düştü.
   "Yazdırdım" yeterli değil; politikanın / ödülün / ortamın gerçekten o
   değeri taşıdığını kontrol et.
6. **Bir sayıyı başka bir düzenekten alıntılama.** Kontrolcü taban çizgisi
   dahil her kolu aynı süreçte, aynı ortamda, aynı tohumlarla ölç.
7. Aynı anda **tek bir Isaac süreci** çalıştır (GPU 8 GB). Uzun işleri
   `nohup … > out/evo/<ad>.log 2>&1 &` ile başlat. Kabuk `fish`; çok satırlı
   komutları `bash -c '…'` veya `#!/usr/bin/env bash` scriptiyle çalıştır.
   `pgrep -f`/`pkill -f` kendi komut satırını da eşler; süreçleri PID ile yönet.
   Değeri `-` ile başlayan argparse argümanlarını `--flag=-0.3` biçiminde ver.

---

## 2. Bağlam: neyi yenmeye çalışıyoruz

Araç: 6 serbestlik dereceli Apollo LM, Isaac Sim / Isaac Lab. Hedef sahne
`orbit_descent` (200 m irtifa, 10–30 m/s yatay hız ile bırakılış). Başarı =
`info["landed_safely"]` (temas anında dikey hız, yanal hız ≤ 1.2 m/s, eğim ve
bacak yükseklik farkı limit içinde).

Ölçülmüş referanslar (Isaac, `orbit_descent`, 16 dalga × 12 ortam = 192
eşleştirilmiş taze bölüm, dalga tohumları 90000–105000):

| kol | iniş | not |
|---|---|---|
| ZEM/ZEV kontrolcüsü (sabit kazanç) | 101/192 = %52.6 | taban çizgisi |
| En iyi sabit kazanç ayarı (PPO ortalama eylemi sabit) | %71.9 | kontrolcüye karşı McNemar 37–0 |
| **Guided-gain PPO** | **172/192 = %89.6** | kontrolcüye karşı 72–1, p=1.6e-20 |
| Uçtan uca PPO (sıfırdan, kontrolcüsüz) | %0 | geçmiş çalışmalar; SAC da çöktü |

Analitik (hızlı, CPU) backend'de `land_handoff` sahnesi: kontrolcü %9.9,
guided-gain PPO %31.2 (en iyi), 200 örnekli kazanç-uzayı kâhini %31.2.

**RL neden zorlandı (ölçüldü):**
- Uzun ufuk: bir bölüm ~800 kontrol adımı (20 Hz). GAE λ=0.99 terminal
  ödülün ağırlığını %0.037'ye indiriyordu.
- 0.05 s karar adımına karşı ~6.3 s tutum zaman sabiti. Tek komutun etkisi
  gürültüde kayboluyor.
- BC'den miras donmuş keşif gürültüsü.
- Yoğun (dense) şekillendirme ödülü politikaları iniş oranının **tersine**
  sıraladı. Sadece terminal sonuç doğru sıralıyor.

**ES neden uygun olabilir:** Bütün bölümü tek bir uygunluk sayısıyla puanlar,
zamansal kredi dağıtımı yapmaz. Gürültüyü parametrelere bir kez ekler, bu
yüzden keşif bölüm boyunca tutarlıdır. Eylem frekansına ve ufuk uzunluğuna
büyük ölçüde duyarsızdır (Salimans ve ark. 2017, arXiv:1703.03864; lineer
politikalarla ARS: Mania ve ark. 2018, arXiv:1803.07055).
**Zayıf yanı:** örnek verimsizliği. Bunu dürüstçe raporla.

---

## 3. Ortam ve araçlar

- Python: `/home/haktan/isaac-env/bin/python` (torch 2.12 + CUDA, Isaac Sim 6.1).
  Analitik scriptler de bununla çalışır. `.venv` yeniden kurulumu yarım kaldı; kullanma.
- Makine: 16 çekirdek, RTX 5060 Laptop 8 GB.
- Ek paket gerekirse (`cma`) sadece `isaac-env`'e kur ve raporda belirt.
  Tercih: OpenAI-ES / sep-CMA-ES'i numpy ile kendin yaz (~100 satır), bağımlılık yok.
- Isaac scriptlerinde `SimulationApp` **lunarsim'in Isaac modülleri import
  edilmeden önce** oluşturulmalı. Bkz. `scripts/train_guided_gain_isaac.py`'nin başı.

### Kullanacağın mevcut arayüzler (değiştirmeden)

```python
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn
from lunarsim.rl.reward import make_apollo_reward_fn, weights_from_overrides
from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv   # Isaac, ham eylem
from lunarsim.rl.guided_gain_vec_env import GuidedGainVecEnv                  # Isaac, kazanç eylemi
from lunarsim.rl.guided_gain_env import GAIN_SETS, GuidedGainEnv              # analitik, kazanç eylemi
from lunarsim.rl.analytic_lander_env import AnalyticLanderEnv                 # analitik, ham eylem
```

Isaac ortamı tam olarak eğitim scriptindeki gibi kurulur:

```python
stage = STAGES_BY_NAME["orbit_descent"]
venv = IsaacLanderVecEnv(
    num_envs=12, tile_fn=make_tile_fn(stage, 80), params=stage.params,
    env_spacing_m=max(s.tile_size_m for s in STAGES) * 1.5,
    seed=7, max_rocks_per_env=0,
    reward_fn=make_apollo_reward_fn(weights_from_overrides(OVERRIDES)))
gvenv = GuidedGainVecEnv(venv, gain_spec=GAIN_SETS["attitude"], hold_steps=20,
                         residual_scale=0.0)
```

- `GuidedGainVecEnv`: eylem `(n_envs, 11)` ∈ [-1, 1]. Kazanç = nominal × oran^a,
  yani **a = 0 tam olarak kontrolcünün kendisi**. Her karar 20 kontrol adımı (1 s)
  tutulur. Gözlem = ham gözlem + 3 ek (`tgo/30, in_final, v_xy/1.2`).
- `IsaacLanderVecEnv`: ham eylem `(n_envs, 4)` ∈ [-1, 1] =
  `[throttle, pitch, roll, yaw]`, 20 Hz. Throttle eşlemesi `lunarsim/rl/action_map.py`
  (kübik, a0 = 0 → T/W ≈ 1.016, yani **hover**).
- `info` alanları: `landed_safely`, `lost_control`, `left_tile`, `timed_out`,
  `impact_state`, `landing_margins`, `terminal_observation`.
- Bitmiş ortamlar otomatik sıfırlanır.

**Eşleştirilmiş değerlendirme (dalga tasarımı)** — `scripts/validate_guided_gain_isaac.py`
içindeki `wave()` fonksiyonunu oku ve yeni dosyaya kopyala:
`gvenv.venv._rng = np.random.default_rng(wave_seed)`, sonra `reset()`, her
ortamdan **sadece ilk** bölümü say. Aynı dalga tohumu, hangi politika uçarsa uçsun,
aynı 12 başlangıç durumunu üretir. Bu, ES için **ortak rastgele sayılar** (CRN)
sağlar: bir nesildeki tüm adaylar aynı başlangıçlarda puanlanır, böylece
uygunluk farkı tohum şansından değil politikadan gelir.

---

## 4. Uygunluk (fitness) fonksiyonu

Yoğun şekillendirme kullanma; bu projede iniş oranıyla ters sıraladığı ölçüldü.
İki uygunluk tanımı, ikisi de mevcut `RewardWeights` alanlarıyla elde edilir
(yeni ödül kodu yazma):

- **F_outcome** (PPO'nun %89.6'yı aldığı ödülün aynısı):
  `reward_scale=0, touchdown_k=0, touchdown_penalty_cap=60, landed_step_k=100`.
  Değerler: iniş +5, başarısız temas 0, zaman aşımı / devrilme −3.
  Uygunluk = dalga(lar) üzerindeki ortalama bölüm getirisi.
- **F_graded** (sürekli terminal, kesik değil):
  `reward_scale=0, touchdown_k=120, touchdown_penalty_cap=60, landed_step_k=100`.
  Başarısız temaslar arasında da "ne kadar sert" farkı verir. **Uçtan uca fazda
  zorunlu:** başlangıçta hiçbir aday inmez, F_outcome'da hepsi berabere kalır
  ve ES hiçbir sinyal almaz.

Uygunluğu ödülden hesaplıyorsan `reward_total_scale` (0.05) çarpanını hesaba kat.
Yeni bir uygunluk eklemeden önce `scripts/diag_reward_rescore.py`'nin
yaklaşımıyla aynı bölümler üzerinde yeniden puanla. Aday uygunluğun iniş
oranıyla **aynı yönde** sıraladığını göster.

---

## 5. Yöntem

### Politikalar (`lunarsim/evo/policies.py`)

1. **Lineer + gözlem normalizasyonu** (ARS tarzı): `a = clip(W · (o − μ)/σ + b, −1, 1)`.
   μ, σ çalışan istatistiklerle güncellenir (ES nesilleri boyunca dondurulmuş
   kopya kullan). Guided-gain için 23 → 11: 264 parametre. Ham kontrol için
   20 → 4: 84 parametre.
2. **Küçük MLP** (tanh, 32×32), uçtan uca faz için. Lineer geri besleme kübik
   throttle eşlemesiyle yetmeyebilir.

Politika numpy ile `predict(obs) -> actions` arayüzünü uygulasın
(`validate_guided_gain_isaac.py`'deki `_Const` sınıfının yaptığı gibi), torch gerekmesin.
Kaydetme formatı `.npz` (parametreler + μ, σ + yapı bilgisi).

**Başlangıç:** Guided-gain'de θ = 0 kontrolcünün kendisidir. Bu meşru bir
başlangıçtır, çünkü PPO da aynı noktadan başladı.

Uçtan uca fazda **hiçbir kontrolcü, demonstrasyon veya BC başlangıcı yok.**
Tek izin verilen ön bilgi: throttle bias'ını hover'ın hafif altına koymak
(ör. b0 = −0.05). Bunu raporda açıkça belirt. Bias = 0 ile de bir kontrol
koşusu yap.

### Optimizer (`lunarsim/evo/es.py`)

- **Birincil:** sep-CMA-ES ya da 264 parametrede tam CMA-ES.
  Popülasyon λ = 4 + ⌊3 ln n⌋ ≈ 20, ayrıca 32 ve 64 dene.
- **İkincil:** OpenAI-ES. Antitetik örnekleme (+ε/−ε çiftleri), sıralama
  tabanlı uygunluk şekillendirme (centered ranks), Adam, ağırlık çürümesi küçük.
  σ_param başlangıç: guided-gain için 0.1–0.3, ham kontrol için 0.3–1.0.
  Bunlar hiperparametre; küçük bir tarama yap ve raporla.
- **CRN:** bir nesildeki tüm adaylar aynı E eğitim dalgası tohumunda uçar.
  Her nesil yeni tohumlar: `200000 + 1000·g`.
- **Elitizm / seçim:** her K nesilde ortalama θ (ve en iyi aday) ayrı bir seçim
  bloğunda (`150000–159000`, 4 dalga) puanlanır. Nihai aday YALNIZCA bu bloktan
  seçilir. Doğrulama dalgaları (90000–105000) seçimde kullanılmaz.

### Bütçe ve adil karşılaştırma

Bütçeyi **karar sayısıyla** say (1 karar = 1 ortam × 1 s). PPO'nun en iyi
checkpoint'i **45 000 karar**da (8 ortam). Raporda iki kıyas olsun:

1. Eşit bütçe: ES 45k kararda nerede?
2. ES'in kendi yakınsama noktası: kaç karar gerekti?

ES muhtemelen eşit bütçede kaybeder. Bu da raporlanacak geçerli bir sonuçtur;
gizleme.

Paralellik: Isaac'te bir dalga 12 ortamı aynı anda uçurur.
- **Basit yol:** aday başına E dalga.
- **Hızlı yol (isteğe bağlı):** bir dalganın 12 yuvasına farklı adaylar koy.
  Bu durumda CRN'yi korumak için her adayın her (dalga, yuva) başlangıcını
  görmesini sağlayan bir rotasyon şeması kur ve doğruluğunu test et.

Faz 0'da gerçek hızı **ölç** ve bütçeyi buna göre planla.

---

## 6. Fazlar

### Faz 0 — Altyapı ve doğruluk (zorunlu, önce)

1. `tests/test_evo_*.py`:
   - Sıfır parametreli guided-gain politikası tam olarak sıfır eylem üretir.
   - Antitetik çiftler simetrik.
   - Sıralama şekillendirme sıfır ortalamalı.
   - `.npz` kaydet/yükle aynı eylemi verir.
2. **Kontrolcü yeniden üretimi:** θ = 0 guided-gain politikası, yeni eşleştirilmiş
   değerlendiricide, 1 dalga (90000) üzerinde kontrolcü koluyla **birebir aynı**
   sonuçları vermeli. Vermiyorsa dur ve sebebini bul.
3. **PPO yeniden üretimi:** `~/lunarsim_artifacts/gg_isaac_orbit_v4_seed21_repro_best_frozen.zip`
   (SB3 PPO, gain-set `attitude`, hold 20) ve `gg_seed21_policy_numpy.npz`
   (aynı politikanın numpy hali) yeni değerlendiricide 2 dalga uçurulsun.
   Sonuçlar `~/lunarsim_artifacts/gg_isaac_orbit_v4_seed21_repro_validation.json`
   içindeki ilk 24 bölümle eşleşmeli.
4. Hız ölçümü: dalga/saniye, karar/saniye. Bütçe tablosunu bu sayılarla yaz.

### Faz 1 — Analitik `land_handoff`, guided-gain ES (hızlı prova)

- `GuidedGainEnv` + çok süreçli havuz (16 çekirdek).
- Analitik doğrulama mantığı için `scripts/validate_guided_gain.py`'yi oku.
  Hedef çıta: PPO %31.2, kâhin %31.2, kontrolcü %9.9.
- Amaç: optimizer hiperparametrelerini Isaac'e zaman harcamadan seçmek.
  Burada da doğrulama tohumlarını seçimde kullanma.

### Faz 2 — Isaac `orbit_descent`, guided-gain ES (ana karşılaştırma)

- Gain-set `attitude`, hold 20, F_outcome. F_graded ayrı bir koşu olarak.
- En az 2 farklı ES tohumu. PPO'nun tek tohumlu sonucunu tekrarlanabilirlik
  açısından geçmek için gerekli.
- Nihai doğrulama: **üç kol, aynı 16 dalga** (90000–105000): kontrolcü,
  PPO (seed21 checkpoint), ES. Raporlanacaklar:
  - her kolun iniş oranı,
  - ES–kontrolcü ve ES–PPO McNemar kesin testleri,
  - bootstrap %95 güven aralığı (dalga bazında yeniden örnekleme).

### Faz 3 — Uçtan uca "saf öğrenme" (kontrolcüsüz)

- Önce analitik `AnalyticLanderEnv` (ham 4 eylem) ile başla. Kontrolcü yok.
  F_graded kullan.
- Önce kolay bir sahne seç, sonra `orbit_descent`. Sahneler:
  `ramp_20m`, `ramp_20m_fast`, `ramp_35m`, `ramp_50m`, `ramp_100m`, `ramp_150m`,
  `orbit_descent`, `land_handoff`.
  Uyarı: geçmiş commit'ler `ramp_100m`, `ramp_150m` ve `ramp_20m_fast`'ın
  süre/bütçe ayarlarıyla çözülemez olduğunu raporladı. Bir sahneyi kullanmadan
  önce kontrolcünün orada inebildiğini ölç (çözülebilirlik kontrolü; kontrolcü
  eğitime girmez).
- Bir müfredat kullanıyorsan aşamalar arası θ aktarımını açıkça raporla.
- Eylem tutma (action repeat) bir hiperparametredir. 1 (20 Hz) ve 5 (4 Hz)
  değerlerini dene.
- Isaac'e ancak analitikte sıfırdan büyük, anlamlı bir iniş oranı çıkınca geç.
- Bu fazda kontrolcüyle kıyas ikincil. Asıl soru: **iniş oranı anlamlı biçimde
  %0'dan büyük mü?** (PPO / SAC sıfırdan %0 aldı.) Binom güven aralığıyla raporla.

---

## 7. Bilinen tuzaklar (bu projede hepsi yaşandı)

- **Hover çekicisi:** ham eylemde a0 = 0 hover demektir. Rastgele / sıfır
  politika 60 s boyunca asılı kalıp zaman aşımına girer. Uygunluk berabere
  kalıyorsa sebep budur.
- **Karo kenarından kaçış istismarı:** `left_tile` bölümü keser. Kenar cezası
  artık canlı (`_EnvView.tile` düzeltildi), ama ES bunu da bulabilir.
  Her raporda sonlanma nedenlerinin dağılımını ver: iniş, çarpma, zaman aşımı,
  devrilme, karo dışı.
- **Etiket hatası geçmişi:** eskiden çarpmalar iniş sayılıyordu. Sadece
  `info["landed_safely"]` kullan; kendi iniş tanımını yazma.
- **İrtifa:** `state["z"]` gövde merkezidir (+3.52 m). Kendi teşhislerinde
  `info["altitude_m"]` veya `GuidedGainVecEnv`'in yaptığı gibi ayak altı açıklığı kullan.
- **Ara-tutma (mid-hold) sonlanma:** `GuidedGainVecEnv` bunu doğru ele alır.
  Ham ortamda kendi eylem tekrarını yazıyorsan, biten ortamın kalan alt adımlarını
  at ve bir sonraki bölümün başlangıcını yeme. Gerekçesi `guided_gain_vec_env.py`'nin
  docstring'inde.
- **Küçük örneklem yanılgısı:** 12–48 bölümle iki aday sıralanamaz
  (±10–14 puan). Nesil içi sıralama CRN ile yapılır. Nihai iddia SADECE
  192 eşleştirilmiş doğrulama bölümünden yapılır.
- **Seçim yanlılığı:** en iyi checkpoint'i seçtiğin blokta raporlama.
- **Sessiz bayrak:** bkz. kural 5.

---

## 8. Teslimatlar

- Kod: `lunarsim/evo/`, `scripts/evo_*.py`, `tests/test_evo_*.py`.
  `pytest tests/test_evo_*.py` geçmeli. Mevcut testler de bozulmamalı; çalıştırıp kontrol et.
- Sonuçlar: `out/evo/<koşu>/`
  - `curve.csv`: nesil, toplam karar, ortalama uygunluk, seçim bloğu iniş oranı, σ.
  - `best_on_selection.npz`
  - `validation.json`: her kolun bölüm bazında iniş vektörleri, dalga tohumları, ayarlar.
  - Bunların kopyası `~/lunarsim_artifacts/evo/` içinde.
- Rapor: `EVO_RESULTS.md` (yeni dosya). İçeriği:
  1. Her fazın sonuç tablosu:
     `kol | iniş/192 | % | 95% GA | McNemar (b–c, p) | karar bütçesi | duvar saati`.
  2. Eşit bütçe kıyası ve ES yakınsama eğrisi.
  3. Sonlanma nedeni dağılımları.
  4. Uçtan uca fazda en iyi politikanın davranışı: temas hızları, yanal hız, eğim.
  5. Ne işe yaramadı, ölçümle.
  6. Hiperparametreler ve tam yeniden üretim komutları.
- Bir fazın sonucu olumsuzsa da raporla. "ES eşit bütçede PPO'nun X puan
  gerisinde" bir sonuçtur. Bir mekanizmayı ölçmeden "işe yaramadı" deme;
  önce doğru çalıştığını (Faz 0 testleri gibi) göster.

## 9. Başarı ölçütleri

| Faz | Asgari | İyi |
|---|---|---|
| 0 | θ = 0 kontrolcüyü, PPO npz PPO'yu birebir yeniden üretir | — |
| 2 | ES kontrolcüyü anlamlı geçer (McNemar p < 0.05, 192 bölüm) | ES, PPO ile istatistiksel olarak ayırt edilemez ya da onu geçer |
| 3 | Uçtan uca ES, analitikte %0'dan anlamlı büyük iniş oranı | Isaac `orbit_descent`'te anlamlı iniş oranı; kontrolcüyü (%52.6) geçerse büyük sonuç |

## 10. Okuma sırası (başlamadan önce)

1. `handover.md` bölüm 0 (son durum) ve "What NOT to do".
2. `lunarsim/rl/guided_gain_env.py` ve `lunarsim/rl/guided_gain_vec_env.py` docstring'leri.
3. `scripts/train_guided_gain_isaac.py`, `scripts/validate_guided_gain_isaac.py`.
4. `lunarsim/rl/reward.py`: `RewardWeights`, `_terminal_reward`, `landed_step_k`.
5. `report/lunarsim_problem_katalogu.pdf`: projede yaşanan 139 problem ve
   denenen yöntemler. Aynı hataları tekrarlamamak için göz at.
