# GPU serverdə işləmək — addım-addım təlimat

Bu təlimatda Xəttatı AI Academy-nin GPU serverində necə qurub train edəcəyiniz yazılıb.
Əmrləri olduğu kimi kopyalayıb **JupyterLab terminalına** yapışdırın.

> ⚠️ `team14.conf` və `README-team14.md` fayllarında VPN açarı və token var. Onları heç vaxt GitHub-a
> yükləməyin və komandadan kənar heç kimə göndərməyin.

---

## ⭐ Ən asan yol: hər şeyi bir əmrlə gecəyə buraxmaq

```bash
cd ~/Xettat && git pull
bash slurm/pipeline.sh
```

Bu əmr bütün mərhələləri ardıcıl işə salır: data → BiLSTM train → qiymətləndirmə → Transformer train → qiymətləndirmə.
Hər mərhələ bitəndə növbətini özü növbəyə qoyur. Bundan sonra VPN-i söndürə və kompüteri bağlaya bilərsiniz.

Səhər vəziyyəti belə yoxlayın:

```bash
squeue --me              # hazırda nə işləyir
cat .pipeline_queue      # hələ qalan mərhələlər (boşdursa, hamısı bitib)
ls -t logs | head        # son log faylları
```

Bir mərhələ xəta ilə bitsə, zəncir orada dayanır. Ən son log faylını Claude-a göndərin.
Data artıq hazırdırsa, `bash slurm/pipeline.sh --skip-data` yazın.

---

## 0. Hər dəfə: serverə qoşulmaq

1. Kompüterdə **WireGuard** proqramını açın, `team14` tunelini seçib **Activate** basın.
2. Brauzerdə `README-team14.md` faylındakı `http://10.9.0.1:8080/team14/?token=...` linkini açın.
3. JupyterLab-da **File → New → Terminal** seçin.

## 1. Bir dəfə: kodu serverə gətirmək və quraşdırmaq

```bash
cd /data/teams/team14
git clone https://github.com/Hasanov57/Xettat.git
cd Xettat
bash scripts/setup_server.sh
```

Repo private-dirsa, `git clone` sizdən istifadəçi adı və şifrə istəyəcək. Şifrə yerinə GitHub
**Personal Access Token** yazmaq lazımdır (GitHub → Settings → Developer settings → Tokens).

`setup_server.sh` əmri PyTorch-u (CUDA 12.8) və digər paketləri quraşdırır. Bu əmr bir neçə dəqiqə çəkir.

## 2. Bir dəfə: hər şeyin işlədiyini yoxlamaq

```bash
sbatch slurm/test_gpu.sbatch
squeue --me            # job-un vəziyyəti (PD = gözləyir, R = işləyir)
```

Job bitəndən sonra nəticəyə baxın:

```bash
ls logs/
cat logs/xettat-test-*.out
```

Sonda `... passed` yazısı görməlisiniz. `failed` yazısı varsa, faylın məzmununu Claude-a göndərin.

## 3. Bir dəfə: datanı hazırlamaq

```bash
sbatch slurm/prepare_data.sbatch
```

Bu job GPU istifadə etmir. Azərbaycan Vikipediyasını yükləyir, cümlələrə bölür, train/val/test
fayllarını yaradır və baseline-ların nəticələrini hesablayır. Təxminən 20–60 dəqiqə çəkir.
Bitəndən sonra nəticələrə baxın:

```bash
cat data/processed/stats.json
cat logs/xettat-data-*.out | tail -5
```

> Job "GPU tələb olunur" xətası ilə qəbul olunmasa, `slurm/prepare_data.sbatch` faylında
> `# If the cluster refuses ...` sətrinin altındakı `#SBATCH --gres=gpu:1g.20gb:1` sətrini aktivləşdirin.

## 4. Modeli train etmək

```bash
sbatch slurm/train.sbatch configs/bilstm.yaml bilstm_v1
```

- Hər dəfə yalnız **bir** train job-u işləyə bilər (team limiti). İkinci job növbədə gözləyəcək.
- Job 8 saata yaxınlaşanda özünü dayandırır, vəziyyəti saxlayır və **avtomatik** yenidən növbəyə düşür.
- Train gedişatına canlı baxmaq üçün:

```bash
squeue --me
tail -f logs/xettat-train-<JOBID>.out        # çıxmaq üçün Ctrl + C
tail -3 runs/bilstm_v1/metrics.jsonl
```

`val_word_acc` sözlərin düzgün bərpa olunma faizidir (1.0 = 100%).

Sonra eyni qaydada Transformer modelini işə salın:

```bash
sbatch slurm/train.sbatch configs/transformer.yaml transformer_v1
```

## 5. Qiymətləndirmək

```bash
sbatch slurm/evaluate.sbatch bilstm_v1 val
```

> **Test split** (`... bilstm_v1 test`) yalnız **sonda, bir dəfə** işə salınır. Bu kursun qaydasıdır.

## 6. Nəticələri GitHub-a göndərmək

Böyük fayllar (data, checkpoint-lər) `.gitignore` ilə bloklanıb. Ona görə yalnız kiçik log faylları
gedəcək:

```bash
git add runs/
git commit -m "runs: bilstm_v1 results"
git push
```

## Faydalı əmrlər

| Əmr | Nə edir |
|---|---|
| `squeue --me` | Komandanın job-ları |
| `scancel JOBID` | Job-u dayandırır |
| `sacct -X -S 2026-11-25` | Bitmiş job-lar və onların statusu |
| `du -sh /data/teams/team14` | İstifadə olunan disk yeri (limit 180 GB) |
| `git pull` | GitHub-dakı son kodu serverə gətirir |

## Problem olduqda

- **Job uzun müddət `PD` vəziyyətində qalır.** `squeue --me` əmrinin REASON sütununa baxın.
  - `Resources` və ya `Priority`: sadəcə növbədir, gözləyin.
  - `AssocGrpGRES`: komandanın başqa bir job-u artıq işləyir.
- **`OUT_OF_MEMORY` xətası.** Konfiq faylında `batch_size`-ı yarıya endirin.
- **Başqa xəta.** `logs/...out` faylının son 30 sətrini Claude-a göndərin.
