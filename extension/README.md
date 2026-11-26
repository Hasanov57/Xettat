# Xəttat — Chrome extension

İstənilən saytda ingilis klaviaturası ilə yazılmış mətni düzgün Azərbaycan yazısına çevirir.
Model **brauzerin içində** işləyir (onnxruntime-web, WebAssembly): mətn kompüterdən çıxmır.

## İstifadə

- Mətn sahəsinə klikləyin → **Ctrl + Shift + Space**. (Qısayolu `chrome://extensions/shortcuts` səhifəsində dəyişmək olar.) Mətnin bir hissəsi seçilibsə, yalnız o hissə düzəlir.
- Və ya sağ klik → **Xəttat ilə düzəlt**.
- Və ya brauzerin yuxarı panelindəki Xəttat ikonuna basın → mətni yapışdırın → **Düzəlt**.
- Geri qaytarmaq: **Ctrl + Z**.

## Modeli əlavə etmək (bir dəfə)

Model faylları repoda saxlanılmır (böyükdür). Train bitəndən sonra:

```bash
python -m xettat.export --ckpt /data/teams/team14/checkpoints/bilstm_v1/best.pt --out exported/bilstm_v1
python scripts/fetch_ort.py
python scripts/pack_extension.py --model exported/bilstm_v1
```

Bu əmrlərdən sonra `extension/model/` və `extension/vendor/` qovluqları yaranır.

## Chrome-a yükləmək

1. Chrome-da `chrome://extensions` ünvanını açın.
2. Sağ yuxarıda **Developer mode** düyməsini aktiv edin.
3. **Load unpacked** basın və bu `extension` qovluğunu seçin.
4. Xəttat ikonunu panelə bərkidin (🧩 → 📌).

## Server rejimi

Ayarlarda **Server** seçilərsə, mətn Docker API-yə göndərilir (daha böyük, daha dəqiq model).
API-ni işə salmaq üçün kök qovluqdakı `README.md`-yə baxın.

## Məhdudiyyətlər

- `chrome://` səhifələrində və Chrome Web Store-da extension-lar işləyə bilmir.
- Mətni yerində dəyişmək mümkün olmayan saytlarda (məs. Google Docs, başqa domendən gələn redaktor çərçivələri) Xəttat popup-ı açır və düzəldilmiş mətni orada göstərir.
