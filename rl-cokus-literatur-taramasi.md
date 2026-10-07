# Literatür Taraması — SAC'ta eğitim ortası çöküşü ve RL ile Ay'a iniş

**Tarih:** 2026-10-06
**Kapsam:** Bizim ölçtüğümüz somut probleme literatürdeki teşhis ve çözümler:
96 bölümde %22.9 ölçülmüş bir politika, 600k adım daha eğitilince beş ardışık
snapshot'ta 0/24'e düşüyor; gamma, öğrenme oranı, entropi katsayısı ve tekrar
oynatma oranı değişiklikleri sonucu değiştirmiyor.
**Kaynaklar:** arXiv MCP, OpenAlex MCP, paper-search MCP. (Semantic Scholar MCP
rate-limit'e takıldı, onun yerine OpenAlex ile çapraz doğrulama yapıldı.)
**Ayrıntılı analiz:** `rl-cokus-yontem-ve-gap-detay.md`

---

## 0. Tek paragraflık cevap

Çöküşün literatürdeki adı var: **plasticity loss** (plastisite kaybı) ve onun
bir alt biçimi olan **primacy bias**. Ve kritik nokta şu: alanın derleme
makalesi, plastisite kaybının **aşırı tahmin yanlılığına yol açtığını** söylüyor
— tersi değil. Biz Q şişmesini görüp ona saldırdık; o bir **belirti**. Bu, dört
müdahalemizin neden başarısız olduğunu ve öğrenme oranını düşürmenin neden
Q'yu durdurup çöküşü durdurmadığını tam olarak açıklıyor: belirtiyi tedavi
ettik, hastalığı değil.

---

## 1. Trend analizi

**2022-2026 arası net bir araştırma dalgası var ve tam bizim problemimiz üzerine.**
Zaman çizgisi:

| yıl | dönüm noktası |
|---|---|
| 2022 | Nikishin et al., *The Primacy Bias in Deep RL* — olgu adlandırılıyor, çözüm olarak **periyodik kısmi reset** öneriliyor |
| 2023 | Sokar et al., *The Dormant Neuron Phenomenon* (ReDo) — ölçülebilir bir metrik (uyuyan nöron oranı) ve hedefli bir çözüm; Abbas et al. *Loss of Plasticity in Continual Deep RL* |
| 2023 | Ma et al., *Revisiting Plasticity in Visual RL* — hangi modülün (aktör mü kritik mi) ve hangi eğitim aşamasının kritik olduğu |
| 2024 | Klein et al. **derleme**: 50+ azaltma stratejisi, ilk kapsamlı taksonomi; Lyle et al. *Normalization and effective learning rates* |
| 2025-2026 | Çözüm tarafı çeşitleniyor: kısmi/kalibre resetler, spektral müdahaleler, ağ büyütme, seyrekleştirme |

**Terminolojik kayma:** "overestimation bias" merkezli 2018-2021 anlatısından,
**"plasticity / expressivity loss"** merkezli 2022+ anlatısına geçiş var. Aşırı
tahmin artık bağımsız bir hastalık değil, plastisite kaybının bir **sonucu**
olarak konumlanıyor (Klein et al. 2024 derlemesi bunu açıkça yazıyor).

**Bizim için anlamı:** 2018-2021 çerçevesinde çalışıyormuşuz. `n_critics`,
gamma, TD3-tarzı min-ensemble hepsi o çerçevenin araçları.

**İniş tarafında bambaşka bir trend var** (bkz. Bölüm 4.4): Ay'a iniş rehberlik
literatürünün ezici çoğunluğu **RL kullanmıyor**; optimal kontrol çözümlerinden
(Pontryagin, dışbükey optimizasyon) üretilen veri setleriyle **gözetimli**
ağ eğitiyor. RL kolu büyük ölçüde tek bir grubun (Gaudet/Linares/Furfaro) işi.

---

## 2. En kritik bulgu: teşhisimiz ters

Klein et al. (2024) derlemesinin özetinden, birebir:

> "Loss of plasticity causes performance plateaus and contributes to scaling
> failures, **overestimation bias**, and insufficient exploration."

Bizim ölçüm zincirimizle yan yana koyalım:

| bizim gözlemimiz | literatürdeki yeri |
|---|---|
| Q 151 → 406 şişiyor | plastisite kaybının **belirtisi** |
| gamma Q'yu %14 düşürdü, fayda yok | belirtiye müdahale |
| lr 5e-5 Q şişmesini **tamamen durdurdu**, çöküş sürdü | belirti kesildi, hastalık sürdü — tam beklenen sonuç |
| çöküş her warm-start'tan ~100k adım sonra, tüm düğmelere duyarsız | plastisite kaybının imzası (hiperparametre değil, kapasite sorunu) |
| politika güvenli/eylemsiz davranışa (hover) kaçıyor | "insufficient exploration" + ifade gücü kaybı |

`lr 5e-5` deneyimiz bu tezin en güçlü kanıtı: Q'yu durdurup çöküşü durduramamak,
ikisinin ayrı şeyler olduğunu gösteriyor.

### Neden bizim kurulumumuz plastisite kaybı için ideal zemin

1. **Warm-start + bellek sıfırlama = ders kitabı primacy bias.** Nikishin et al.
   (2022): ajanlar **erken** etkileşimlere aşırı uyum sağlayıp sonraki kanıtı
   yok sayıyor. Biz her koşuda belleği sıfırlıyoruz, kritik dar bir erken
   pencereden kendini yeniden kuruyor.
2. **Uzun süre eğitilmiş başlangıç politikası.** *Policy Plasticity Matters in
   Offline-to-Online RL* (2026): statik veri üzerinde uzun süre optimizasyon,
   politikanın çevrimiçi uyum yeteneğini düşürüyor. Bizim warm-start
   checkpoint'lerimiz 1.5M adım görmüş.
3. **Müfredat = kasıtlı durağan olmayan dağılım.** Abbas et al. (2023) tam bunu
   ölçüyor: durağan olmayanlık derecesi arttıkça plastisite kaybı artıyor. Bizim
   basamak geçişleri bunu bilerek yapıyor.
4. **Yüksek tekrar oynatma oranı.** 16 ortamla 1:1, yani benzersiz geçiş başına
   16 güncelleme — plastisite literatürünün "replay ratio" ekseninin yüksek ucu.

---

## 3. Gap analizi (özet — ayrıntı: `rl-cokus-yontem-ve-gap-detay.md`)

| # | Boşluk | Bizim için anlamı |
|---|---|---|
| G1 | Plastisite literatürü neredeyse tamamen **benchmark** ortamlarında (Atari, DMC, MetaWorld, MinAtar). Fizik-motoru tabanlı, terminal-kısıtlı **görev** ortamlarında (iniş, manipülasyon) ölçüm yok | Bizim ortam literatürün test etmediği rejimde; metrikleri (uyuyan nöron) kendimiz ölçmek zorundayız |
| G2 | **Müfredat + plastisite kesişimi boş.** Müfredat RL literatürü görev sıralamasına, plastisite literatürü tek görevde bozulmaya bakıyor; basamak geçişinin plastisiteye ne yaptığını ölçen iş yok | Bizim en çok acı çektiğimiz nokta tam bu kesişim |
| G3 | Reset yöntemleri **peak performansı feda ediyor** ve bunu itiraf ediyorlar (AltNet 2026: "resets come at the cost of a temporary drop in performance, which can be dangerous"). Güvenlik-kritik görevlerde reset sonrası geçici çöküşü yönetme çalışması yok | İniş görevinde reset sonrası geçici bozulma kabul edilebilir (simülasyon), ama snapshot disiplinimiz şart |
| G4 | Ay'a iniş RL literatürü **ince** ve çoğu tek grup; SAC ile iniş neredeyse yok (PPO/meta-RL baskın) | SAC seçimimiz literatürde desteklenmemiş bir tercih — gözden geçirilebilir |
| G5 | Gözetimli rehberlik ağları (G&CNET, OINN) optimal kontrolü taklit ediyor ama **arazi/engel kaçınma ve hazard** yok; RL kolu ise optimal garanti vermiyor | İkisini birleştirmek (taklit + RL ince ayar) açık bir boşluk ve bizim elimizde iki parça da var |

---

## 4. Yöntem karşılaştırması (özet — ayrıntı detay dosyasında)

### 4.1 Plastisite kaybına karşı yöntem aileleri

| aile | temsilciler | avantaj | dezavantaj / itiraf |
|---|---|---|---|
| **Periyodik tam/kısmi reset** | Nikishin+22, Reset Deep Ensemble (Kim+23), AltNet (2026) | Basit, algoritmadan bağımsız, tutarlı kazanç | Reset sonrası **geçici performans düşüşü**; güvenlik-kritikte riskli |
| **Nöron düzeyinde geri dönüşüm** | ReDo (Sokar+23), KNIFE (2026), Calibrated Partial Resets (2026) | Ölçülebilir metrik (uyuyan nöron), hedefli | Tam reset "policy collapse"a yol açabiliyor; CPR bunu kalibre ederek çözdüğünü iddia ediyor |
| **Normalizasyon / regularizasyon** | Lyle+24, SimBa (2024) | Derlemeye göre **genel regularizasyon alan-özel müdahaleleri geçiyor** | Normalizasyon, parametre normu büyüdükçe **etkin öğrenme oranını** düşürüyor (gizli yan etki) |
| **Ağ büyütme / seyrekleştirme** | Neuroplastic Expansion (2024), Sparsity-Driven (2025) | Kapasiteyi dinamik yönetiyor | Karmaşık, mimari değişikliği gerektiriyor |
| **Spektral müdahale** | SingularClip (2026), SPHERE (2026) | Yeni ve teorik temelli | Çok yeni, bağımsız doğrulama yok |

**Derlemenin en pratik cümlesi:** genel regularizasyon teknikleri, alan-özel
müdahaleleri genellikle **geçiyor**. Yani egzotik çözüm aramadan önce
normalizasyon/regularizasyon denenmeli.

### 4.2 Bizim bang-bang bulgumuz: literatür "düzeltmeye kalkma" diyor

Ölçtüğümüz şey (politika %95 doygun, `[-1,1,-1,1]`, deterministik ve stokastik
çağrılar birebir aynı) literatürde birebir incelenmiş:

**Shamass (2026), *Unthrottling the Tanh Jacobian in SAC: A Negative Result on
Bang-Bang Control and MetaDrive*:**
- Mekanizma bizim ölçtüğümüzle aynı: `∂a/∂u = 1 - a²`, `|a| → 1` iken sıfıra
  gidiyor, yani aktör tam uçlarda kritik sinyalinden mahrum kalıyor.
- Yazar bunu düzeltmeyi denemiş (aktör kaybına ek terim). **Sonuç negatif:**
  kapısız müdahale politikayı %99 doygunluğa itip getiriyi -31.6'dan
  **-195.5**'e çökertmiş; kapılı sürüm de işe yaramamış.
- Ayrıca: otomatik ayarlı entropi katsayısı müdahaleye **karşı yükselmiş**.
- Kapanış cümlesi bizim için birebir geçerli: *"Saturating a bound is not the
  same as solving a problem whose optimum lives on that bound."*

**Bizim için sonuç:** bang-bang doygunluğunu "kusur" sayıp düzeltmeye
kalkmadığımız iyi olmuş — bu bağımsız bir doğrulama. Ayrıca bizim ölçümümüz de
(doygunluk iyi politikada da %95) aynı yöne işaret ediyordu. İlgili ikinci
çalışma: Chen et al. (2024), tanh dönüşümünün dağılım kaymasına yol açtığını
gösteriyor.

### 4.3 Aşırı tahmine karşı yöntemler — bizim dört deneyimizle çelişen hiçbir şey yok

Literatür `n_critics`/min-ensemble'ı aşırı tahmin için öneriyor, ama hiçbir
kaynak bunun **plastisite kaynaklı** çöküşü çözdüğünü iddia etmiyor. Derleme
tam tersini söylüyor (aşırı tahmin plastisite kaybının sonucu). Yani
`--n-critics 5` koşumuzun fayda etmemesi literatürle **tutarlı** olur.

### 4.4 İniş rehberliği: literatürün fiilen yaptığı şey RL değil

Bulduğum Ay'a iniş rehberlik çalışmalarının çoğunluğu **gözetimli**:

| çalışma | yaklaşım |
|---|---|
| Wang, Chen, Li (2024) — *Fuel-optimal powered descent guidance ... using NNs* | Pontryagin Minimum Principle ile optimal yörünge veri seti üret → ağ eğit |
| Wang, Chen, Lu, Li (2024) — *NN-Based Optimal Guidance for Lunar Vertical Landing* | Aynı aile, terminal tutum kısıtı dahil |
| Origer & Izzo (2024) — *Guidance & Control Networks (G&CNETs)* | Optimal kontrol politikasını temsil eden ağ, Neural ODE ile doğruluk artırma |
| Wang (2026) — *Optimality-Informed Neural Networks* | Enerji-optimal, serbest son zamanlı iniş; sınırlı zarf içinde herhangi bir başlangıçtan |
| Shen et al. (2022) | Dışbükey optimizasyon + ağ, gerçek zamanlı |
| **Gaudet, Linares, Furfaro (2018)** — *Deep RL for Six DoF Planetary Powered Descent* | **RL kolu** — alanın kanonik RL çalışması |
| Gaudet & Furfaro (2021) | RL + stabilize arayıcı, iniş alanı tespiti |

**Bizim için en pratik sonuç:** elimizde `orbit_descent`'te 22/24 (%92) inen bir
ZemZev kontrolcüsü var. Literatürün standart yolu, o çözümü **taklit etmek**
(gözetimli / davranış klonlama / DAgger / G&CNET tarzı), sonra gerekiyorsa RL
ile ince ayar yapmak. Biz SAC'a aynı çözümü sıfırdan yeniden keşfettirmeye
çalışıyoruz — literatürde desteği zayıf, zor yol.

---

## 5. Neyi eksik yapıyoruz — somut liste

1. **Uyuyan nöron oranını hiç ölçmedik.** ReDo'nun metriği bu ve plastisite
   kaybının doğrudan göstergesi. Elimizdeki iyi (%22.9) ve çökmüş (%0)
   checkpoint'leri karşılaştırmak için tek ihtiyacımız bir ileri geçiş. **En
   ucuz ve en bilgilendirici sıradaki ölçüm bu.**
2. **Periyodik kısmi reset hiç denenmedik.** Literatürün 2022'den beri birinci
   çözümü. Özellikle *Calibrated Partial Resets* (2026) doğrudan "policy
   collapse"u hedefliyor ve 400M adımda çöküşten kaçınan tek yöntem olduğunu
   iddia ediyor.
3. **Normalizasyon katmanı yok.** Derleme, genel regularizasyonun alan-özel
   müdahaleleri geçtiğini söylüyor; SB3'ün varsayılan MlpPolicy'sinde
   LayerNorm yok. Lyle et al. (2024) ayrıca normalizasyonun aşırı tahminle de
   savaştığını belirtiyor — bir taşla iki kuş.
4. **Kritik modül ayrımını test etmedik.** Ma et al. (2023), hangi modülün
   (aktör/kritik/encoder) plastisite açısından kritik olduğunun fark ettiğini
   gösteriyor. Biz hep ikisini birlikte eğittik.
5. **Taklit yolunu kapattık.** Kullanıcı kararıyla belleğe demo koymuyoruz
   ("kendi bulsun"). Bu meşru bir tercih ama literatürün iniş alanındaki ana
   yolu bu — yeniden değerlendirmeye değer bir karar noktası.

---

## 6. Önerilen sıradaki adımlar (maliyet sırasına göre)

| # | Adım | Maliyet | Dayanak |
|---|---|---|---|
| 1 | İyi ve çökmüş checkpoint'lerde **uyuyan nöron oranını** ölç | dakikalar, GPU'suz | Sokar+23 |
| 2 | **LayerNorm** ekle (`policy_kwargs` ile) ve bir koşu | bir koşu | Klein+24 derleme, Lyle+24 |
| 3 | **Periyodik kısmi reset** (son katman / düşük fayda nöronları), snapshot disipliniyle | bir-iki koşu | Nikishin+22, CPR 2026 |
| 4 | Warm-start'ta **belleği koru** (`save/load_replay_buffer`) | kod + bir koşu | primacy bias doğrudan |
| 5 | ZemZev kontrolcüsünden **davranış klonlama** + RL ince ayar | daha büyük iş | iniş literatürünün ana yolu (Bölüm 4.4) |

Not: handover.md'deki "sıradaki deney" olarak belleği korumayı yazmıştım; bu
tarama onu **4. sıraya** düşürüyor, çünkü 1-3 daha ucuz ve literatürde daha
güçlü destekli.

---

## 7. Kaynaklar

Plastisite / primacy bias:
1. Nikishin, Schwarzer, D'Oro, Bacon, Courville (2022). *The Primacy Bias in Deep Reinforcement Learning*. arXiv:2205.07802
2. Sokar, Agarwal, Castro, Evci (2023). *The Dormant Neuron Phenomenon in Deep Reinforcement Learning* (ReDo). arXiv:2302.12902
3. Abbas, Zhao, Modayil, White, Machado (2023). *Loss of Plasticity in Continual Deep Reinforcement Learning*. arXiv:2303.07507
4. Klein, Luther, McAuliffe, Miklautz, Plant, Tschiatschek (2024). *Plasticity Loss in Deep Reinforcement Learning: A Survey*. arXiv:2411.04832
5. Ma, Li, Zhang, Liu, Wang et al. (2023). *Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages*. arXiv:2310.07418
6. Lyle, Zheng, Khetarpal, Martens, van Hasselt, Pascanu, Dabney (2024). *Normalization and effective learning rates in reinforcement learning*. arXiv:2407.01800
7. Kim, Shin, Park, Sung (2023). *Sample-Efficient and Safe Deep RL via Reset Deep Ensemble Agents*. arXiv:2310.20287
8. Maheshwari, Raisbeck, da Silva (2026). *AltNet: Addressing the Plasticity-Stability Dilemma in RL*. arXiv:2512.01034
9. McCutcheon, Chatzaroulas, Fallah (2026). *Calibrated Partial Resets: Preventing Policy Collapse in Continual RL*. arXiv:2607.24996
10. Liu, Obando-Ceron, Courville, Pan (2024). *Neuroplastic Expansion in Deep RL*. arXiv:2410.07994
11. Lee, Cho, Kim, Gwak, Kim et al. (2023). *PLASTIC: Improving Input and Label Plasticity*. arXiv:2306.10711
12. Dohare, Hernandez-Garcia, Rahman, Mahmood, Sutton (2023). *Maintaining Plasticity in Deep Continual Learning*. arXiv:2306.13812
13. Kastner, De La Vega, Farahmand (2026). *SingularClip: Preventing Spectral Collapse to Maintain Plasticity*. arXiv:2608.18319
14. Todorov, Cardenas-Cartagena, Cunha, Zullich, Sabatelli (2025). *Sparsity-Driven Plasticity in Multi-Task RL*. arXiv:2508.06871
15. Huang, Qing, Chi, Kong, Zou (2026). *Policy Plasticity Matters in Offline-to-Online RL*. arXiv:2609.33127
16. Falzari & Sabatelli (2025). *Fisher-Guided Selective Forgetting: Mitigating the Primacy Bias*. arXiv:2502.00802
17. Lee, Hwang, Kim, Kim, Tai et al. (2024). *SimBa: Simplicity Bias for Scaling Up Parameters in Deep RL*. arXiv:2410.09754

SAC / tanh doygunluk:
18. Haarnoja, Zhou, Abbeel, Levine (2018). *Soft Actor-Critic*. arXiv:1801.01290
19. Shamass (2026). *Unthrottling the Tanh Jacobian in SAC: A Negative Result on Bang-Bang Control and MetaDrive*. arXiv:2609.09478
20. Chen, Zhang, Wang, Xu, Shen, Zhang (2024). *Rethinking Soft Actor-Critic in High-Dimensional Action Spaces: The Cost of Ignoring Distribution Shift*. arXiv:2410.16739
21. Xu, Hu, Liang, McAleer, Abbeel, Fox (2021). *Target Entropy Annealing for Discrete Soft Actor-Critic*. arXiv:2112.02852

İniş rehberliği:
22. Gaudet, Linares, Furfaro (2018). *Deep Reinforcement Learning for Six Degree-of-Freedom Planetary Powered Descent and Landing*. arXiv:1810.08719
23. Gaudet & Furfaro (2021). *Integrated Guidance and Control for Lunar Landing using a Stabilized Seeker*. arXiv:2112.08540
24. Wang, Chen, Li (2024). *Fuel-optimal powered descent guidance for lunar pinpoint landing using neural networks*. arXiv:2404.06722
25. Wang, Chen, Lu, Li (2024). *Neural-Network-Based Optimal Guidance for Lunar Vertical Landing*. arXiv:2402.12920
26. Origer & Izzo (2024). *Closing the gap: Optimizing Guidance and Control Networks through Neural ODEs*. arXiv:2404.16908
27. Wang (2026). *Optimality-Informed Neural Networks for Lunar Landing Trajectory Optimization*. arXiv:2607.02741
28. Shen, Zhou, Yu (2022). *Real-time computational powered landing guidance using convex optimization and neural networks*. arXiv:2210.07480
29. Capolupo & Rinalducci (2023). *Descent & Landing Trajectory and Guidance Algorithms with Divert Capabilities for Moon Landing* (ESA Argonaut). arXiv:2305.13846
30. Kumar, Chakrabarti, Rallapalli, Kumar, Kakula (2026). *Real-Time Retargeting Using Controllability Boundary for Chandrayaan-3 Lunar Landing*. arXiv:2605.29412

# ================================================================
# HOW THE CLOSEST PUBLISHED WORK ACTUALLY SHAPES THIS PROBLEM
# Gaudet, Linares, Furfaro -- arXiv:1810.08719
# "Deep RL for Six Degree-of-Freedom Planetary Powered Descent and Landing"
# ================================================================

Same problem class: 6-DOF powered descent to a soft, upright, low-rate touchdown
with position/velocity/attitude/rate limits. They use PPO rather than SAC, but the
reward and value-scaling decisions are the transferable part, and five of them differ
from ours in ways that line up with exactly where this project is stuck.

## 1. The shaping term is a VELOCITY FIELD to track, not a speed limit

```
  v_targ = -v_o * (r_hat/||r_hat||) * (1 - exp(-t_go/tau))
  t_go   = ||r_hat|| / ||v_hat||

           r_hat = r - [0,0,15]      if r_z > 15   else [0,0,r_z]
           v_hat = v - [0,0,-2]      if r_z > 15   else v - [0,0,-1]
           tau   = tau_1 = 20 s      if r_z > 15   else tau_2 = 100 s

  reward term:  alpha * ||v - v_targ||,   alpha = -0.01
```

Above 15 m they aim at a point 15 m over the pad with a target vz of -2 m/s; below
15 m the lateral target velocity is set to ZERO so the descent becomes vertical, with
a target vz of -1 m/s. The magnitude decays as t_go shrinks, which is what makes the
touchdown soft. They note the vertical-descent target has the side effect of keeping
attitude level with small rates -- the attitude criteria come for free from the
velocity field rather than from attitude penalties.

**This is the biggest single difference from ours, and it is the one that matters.**
Our descent term is an ENVELOPE -- a speed limit with zero penalty inside it. Measured
on our own weights: at alt 10 m the envelope is 2.47 m/s and the penalty at
vz = -1.00, -2.00, -2.47 is 0.0000, 0.0000, 0.0000. So inside the envelope the agent
is told nothing about how to descend; it only meets a wall when it dives too fast.
A velocity field has a gradient EVERYWHERE and always names a specific velocity to
hold. That is the difference between "do not cross this line" and "here is what to do",
and it is the same barrier-versus-gradient distinction that the dense-term audit found
in the tilt and angular-rate terms.

## 2. The terminal bonus is SMALLER than the dense shaping

```
  Table 5:  alpha=-0.01  beta=-0.05  gamma=-100  delta=-20  eta=+0.01  kappa=10
```

kappa = 10 for a successful landing. The tracking term at a ~5 m/s velocity error
costs 0.05/step, about 20 over a 400-step episode. So terminal/dense is roughly 0.5 --
the dense shaping is TWICE the terminal.

Ours, measured: terminal 351-450 against a dense shaping sum of 32.4, i.e. 10.8x the
wrong way. After the continuous-terminal change it is 3.7x -- still an order of
magnitude from the published balance. The redesign was in the right direction and far
too timid, and it confirms the correction already recorded in handover.md: lowering the
terminal was only half the change, the dense terms had to come UP.

## 3. eta is a POSITIVE per-step reward, and the paper says why

> "eta is a constant positive term that encourages the agent to keep making progress
> along the trajectory. Since all other rewards are negative, without this term, an
> agent would be incentivized to violate the attitude constraint and prematurely
> terminate the episode to maximize the total discounted rewards received starting
> from the initial state."

eta = +0.01 per step. **We have the exact opposite:** a time PENALTY of
time_k * reward_scale = 400 * 0.00025 = 0.1 per step, negative, and ten times larger
in magnitude than their anti-self-termination bonus. So this project pays the agent to
end the episode early, which is precisely the pathology it then measured: a crash cost
-60 against a timeout's -105, and the RL policy converted 33 of the clone's timeouts
into 23 crashes with the landing count unchanged. The reordering committed earlier
treats the symptom; the sign of the per-step term is the cause.

## 4. The observation carries the velocity ERROR, not raw position

```
  obs = [ v_error, q, omega, r_z, t_go ],   v_error = v - v_targ
```

> "Note that aside from the altitude, the lander translational coordinates do not
> appear in the observation. This results in a policy with good generalization in that
> the policy's behavior can extend to areas of the full state space that were not
> experienced during learning."

Ours feeds raw `x - target_x` and `y - target_y`. Theirs hands the policy an
already-computed guidance error, so the network only has to learn a feedback law on
that error rather than rediscover the entire guidance problem from raw state. Note
this also means their policy cannot overfit to absolute position, which is relevant
to our terrain-seed finding.

## 5. Attitude is penalised only near loss of control

delta = -20 on a hinge `-max(0, q_i - q_mgn_i)` with q_mgn = 5*pi/16 ~ 56 deg, and
gamma = -100 one-shot at q_lim = 7*pi/16 ~ 79 deg. Nothing at all below 56 degrees.

This independently corroborates backing out the dense tilt term: the controller here
tilts 8-13 deg routinely to brake, and a term that charged it at the 15 deg touchdown
criterion would have taxed the one maneuver that kills lateral velocity. The published
design leaves tilt free through the whole working range.

## 6. Value-function scaling

> "For value function parameter learning, we use a heuristic that multiplies the
> rewards accumulated over an episode by a factor of 1-gamma."

Plus: observations scaled with incrementally-updated statistics ("avoiding
discontinuities in the scaling statistics"), actions scaled so max thrust = 1, and
"it is important to ensure that the magnitude of the neural network outputs are
reasonably close to unity."

We do none of the value-side scaling. At gamma=0.995 the factor is 0.005, and our
measured Q values live at -7 to -41 -- nowhere near unity.

## 7. Their terminal bonus IS a step function

kappa is paid only when position, velocity, attitude AND rate are all inside their
limits -- a hard step, exactly the shape this project replaced. That corroborates the
v62 result rather than contradicting it: continuity was not what was broken. A step
terminal works when the dense shaping carries the trajectory, which in their design it
does, at twice the terminal's weight.

## What this prescribes for us, in order of expected effect

  1. Replace the descent ENVELOPE with a velocity-field tracking term. This is the
     one that gives the agent something to climb at every step.
  2. Flip the per-step time penalty to a small POSITIVE alive bonus.
  3. Put v_error (and t_go) in the observation instead of raw target-relative position.
  4. Rebalance so dense shaping exceeds the terminal, rather than 3.7x under it.
  5. Scale the critic's targets by (1-gamma).

1, 2 and 4 are reward-side and cheap. 3 changes the observation, so it invalidates
every existing checkpoint and needs fresh demos -- schedule it deliberately, not as a
side effect.
