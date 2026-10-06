# Ayrıntılı gap ve yöntem analizi — SAC çöküşü ve RL ile iniş

Ana rapor: `rl-cokus-literatur-taramasi.md`. Bu dosya onun tekrarı değil,
derinleştirmesi — özellikle bizim kendi ölçümlerimizle literatürün kesiştiği
ve **çeliştiği** yerler.

---

## A. Bizim ölçümlerimiz literatürün iki ana hipotezini de çürütüyor

Bu, bu taramanın en önemli çıktısı ve ana rapordaki tabloların ötesine geçiyor.

### A.1 Aşırı tahmin hipotezi (2018-2021 çerçevesi)

Dört bağımsız müdahale:

| müdahale | Q'ya etkisi | iniş oranına etkisi |
|---|---|---|
| gamma 0.998 → 0.997 | −%14 | yok |
| tekrar oynatma 1:1 → 1:4 | — | yok |
| öğrenme oranı 3e-4 → 5e-5 | şişme **tamamen durdu** (406 → 222 plato → 175) | yok, yine %0 |
| entropi 0.05 → 0.005 | std 4.5 → 3.2 | yok, %21 → %4 |

Öğrenme oranı deneyi tek başına belirleyici: belirtiyi tamamen kesip hastalığı
durduramamak, ikisinin ayrı olduğunu kanıtlıyor.

### A.2 Plastisite kaybı hipotezi (2022+ çerçevesi)

Klein et al. (2024) derlemesi aşırı tahmini plastisite kaybının **sonucu** olarak
konumlandırıyor, bu da A.1'i güzel açıklıyordu. Ama ölçtük ve tutmadı:

**Uyuyan nöron (ReDo metriği, tau=0.025):** aktör %26.0 (iyi) vs %25.0-28.9
(çökmüş); kritik %12.5 vs %12.8-13.4. **Ayırt etmiyor.**

**Ağırlık normu ve özellik rankı:**

| checkpoint | aktör ‖W‖ | kritik ‖W‖ | rank | srank |
|---|---|---|---|---|
| v24 iyi (%22.9) | 190.7 | 484.8 | 137/256 | 29 |
| v29 çökmüş (lr 3e-4) | 223.5 | 598.3 | 130 | 24 |
| v30 çökmüş (lr 5e-5) | 192.2 | 486.8 | 144 | 31 |

v29 klasik imzayı veriyor (norm ↑, rank ↓) ve plastisite anlatısına uyuyor.
**v30 vermiyor ve yine çöktü.** Norm sabit, rank iyi checkpoint'ten yüksek,
uyuyan nöron sabit — buna rağmen %0.

**Mekanizma analizi:** v30'un aktör normu 190.7 → 192.2, yani ağ neredeyse hiç
değişmemiş; performans %22.9 → %0. Minik parametre değişimi, toptan davranış
değişimi. Bu, **doygun** bir politikanın imzası: `|a| = 1`'de parametre→davranış
haritası basamak fonksiyonu gibi, yani küçük gradyan adımları eylemleri
rafine etmek yerine **uçlar arasında atlatıyor**.

**Yakın ıskalayan çalışma:** Shamass (2026) tam bu Jacobian etkisini
inceliyor (`∂a/∂u = 1 − a²`) ama onu bir **kusur** olarak ele alıp düzeltmeye
çalışıyor ve negatif sonuç alıyor. Bizim ölçümümüz onu farklı konumlandırıyor:
doygunluk kendi başına kötü değil (iyi politikada da %95), ama **optimizasyonu
kırılgan** yapıyor. Bu ayrımı yapan bir çalışma bulamadım — bu gerçek bir boşluk.

**Denenebilir yön:** eylem kutusunu, çalışma noktası tanh sınırında değil
**içeride** olacak şekilde yeniden ölçeklemek. Bizim `action_map.py`'deki expo
eğrisi kullanılabilir bandı %3.7'den %16'ya çıkardı ama politika yine uçlarda
oturuyor; kutunun kendisi fazla geniş olabilir.

---

## B. Gap'lerin ayrıntılı analizi

### G1 — Plastisite literatürü görev ortamlarını test etmiyor

**Kanıt:** Taradığım 17 plastisite makalesinin ortamları: Atari 100k, DeepMind
Control Suite, MetaWorld, MinAtar, SlipperyAnt, çok-görevli MuJoCo. Fizik
motorlu, **terminal kısıtlı** (temas hızı/tutum limitleri olan) bir görev yok.

**Neden böyle:** Benchmark'lar karşılaştırılabilirlik için seçiliyor; terminal
kısıtlı görevlerde başarı ikili (indi/inmedi), bu da plastisite metriklerinin
istatistiğini zorlaştırıyor.

**Bizim için:** metrikleri kendimiz ölçmek zorundayız ve yukarıda ölçtük —
sonuç literatürün beklediği gibi çıkmadı. Bu, G1'in gerçek bir boşluk olduğunun
dolaylı kanıtı.

### G2 — Müfredat × plastisite kesişimi boş

**Kanıt:** Müfredat tarafı (CP-DRL 2025, Causal-Paced DRL) görev sıralamasına
bakıyor, plastisiteden söz etmiyor. Plastisite tarafı (Abbas+23) durağan
olmayanlığı **sentetik** olarak değiştiriyor, müfredat kullanmıyor.

**Neden böyle:** İki topluluk ayrı; müfredat "hangi görev sırası" sorusuna,
plastisite "ağ neden bozuluyor" sorusuna odaklı.

**Bizim için:** Basamak geçişinde hem dağılım hem bellek değişiyor (biz belleği
sıfırlıyoruz). Bunun plastisiteye etkisini ölçen iş yok.

### G3 — Reset yöntemleri peak performansı feda ediyor

**Kanıt:** AltNet (2026) doğrudan itiraf ediyor: "resets come at the cost of a
temporary drop in performance, which can be dangerous". Calibrated Partial
Resets (2026) bunu kalibre ederek çözdüğünü iddia ediyor ve 400M adımda çöküşten
kaçınan **tek** yöntem olduğunu söylüyor.

**Bizim için:** Simülasyonda geçici düşüş sorun değil — ama bizim snapshot
disiplinimiz (her 50k) tam bunun için gerekli, ve ölçümü 96 bölümle yapmak şart.

### G4 — Ay'a iniş RL kolu ince, SAC neredeyse yok

**Kanıt:** RL ile iniş bulduğum çalışmalar: Gaudet/Linares/Furfaro (2018, 6-DoF
powered descent), Gaudet/Furfaro (2021, stabilize arayıcı), Gaudet/Furfaro
(2021, hipersonik terminal — farklı görev). Üçü de aynı grup. SAC kullanan
iniş çalışması bulamadım.

**Bizim için:** SAC tercihimiz literatürde desteklenmiş bir seçim değil. Gaudet
grubu meta-RL/PPO ailesini kullanıyor. Bu, algoritma seçimini gözden geçirmek
için bir gerekçe — ama tek başına yeterli değil, çünkü SAC'ın sürekli kontrolde
genel başarısı güçlü.

### G5 — Gözetimli rehberlik ağları ile RL'in birleşmediği yer

**Kanıt:** G&CNET (Origer & Izzo 2024), OINN (Wang 2026), FOPDG-NN (Wang+24)
optimal kontrolü taklit ediyor — **ama** hiçbiri arazi/engel/hazard ele
almıyor; hepsi sabit iniş noktası ve pürüzsüz yüzey varsayıyor. RL kolu
(Gaudet+) araziyle başa çıkıyor ama optimal garanti vermiyor.

**Bizim için:** Elimizde LiDAR hazard haritalama ve gerçek arazi var, **artı**
inen bir kontrolcü. İkisini birleştirmek (kontrolcüden taklit + arazi için RL
ince ayar) bu boşluğa tam oturuyor ve bizim elimizde iki parça da hazır.

---

## C. Yöntem ailelerinin ayrıntısı

### C.1 Reset ailesi

- **Nikishin+22 (primacy bias):** ajanın bir **kısmını** periyodik sıfırla.
  Atari 100k ve DMC'de tutarlı kazanç. En basit ve en çok atıf alan.
- **Kim+23 (Reset Deep Ensemble):** topluluk üyelerini kaydırmalı sıfırla,
  böylece performans hiç sıfırlanmıyor — güvenlik vurgusu var.
- **ReDo (Sokar+23):** sadece **uyuyan** nöronları geri dönüştür. Ölçülebilir
  metrik getiriyor. **Bizim ölçümümüz bu yolun bize fayda etmeyeceğini söylüyor**
  (uyuyan oran iyi/kötü ayırmıyor).
- **CPR (2026):** ikili reset yerine düşük faydalı nöronları başlangıca doğru
  **kısmen** çek, güç nöron faydasıyla ölçekli. "Policy collapse"u isim olarak
  hedefliyor.
- **AltNet (2026):** plastisite-kararlılık ikilemi; reset sonrası düşüşü
  azaltmaya odaklı.

### C.2 Normalizasyon / regularizasyon ailesi

- **Lyle+24:** normalizasyon katmanları hem kayıp manzarasını hem aşırı tahmini
  iyileştiriyor, **ama** parametre normu büyüdükçe **etkin öğrenme oranı**
  düşüyor — sürekli öğrenmede bu etkin LR'ı çok hızlı sıfıra indirebiliyor.
  Bizim v29'daki norm büyümesi (190.7 → 223.5) tam bu mekanizmaya denk geliyor.
- **SimBa (2024):** basitlik yanlılığı enjekte ederek derin RL'de parametre
  ölçeklemeyi mümkün kılıyor.
- **Derlemenin kararı:** genel regularizasyon, alan-özel müdahaleleri
  **geçiyor**. Bu yüzden önerilen sıramızda LayerNorm birinci.

### C.3 İniş rehberliği aileleri

| aile | temsilciler | avantaj | dezavantaj |
|---|---|---|---|
| Analitik/optimal kontrol | Frenkel & Shaferman (2026), Capolupo (ESA Argonaut 2023) | Optimal garanti, açıklanabilir | Arazi/hazard yok, gerçek zamanlı maliyet |
| Dışbükey + ağ | Shen+22 | Gerçek zamanlı, optimalden sapma küçük | Dışbükeyleştirme varsayımları |
| Optimal yörüngeden gözetimli ağ | Wang+24 ×3, Origer & Izzo 2024, Wang 2026 | **Eğitim kararlı** (gözetimli), optimali taklit ediyor | Veri seti üretimi gerekli; arazi/hazard yok; dağılım dışına çıkınca garanti yok |
| RL | Gaudet+18, Gaudet & Furfaro 21 | Arazi/gürültü/bozucu ile başa çıkıyor, uçtan uca | Optimal garanti yok, eğitim kararsız (**bizim yaşadığımız**) |
| Hibrit (taklit + RL) | — | — | **Boşluk (G5)** |

Gözetimli kolun en çarpıcı avantajı bizim bağlamımızda şu: **eğitim kararlılığı
sorunu yok.** Gözetimli öğrenme çökmüyor. Bizim 600k adımda %22.9 → %0 yaşadığımız
problem o çerçevede ortaya çıkmıyor.
