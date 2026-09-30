# Model seçimi

src/app.py dosyasını Python dosyalarını topladığın klasördeki app.py yerine koy. template/index.html dosyasını proje kökündeki template/index.html yerine koy. Model dosyaların değişmez. app.py dosyasının bir üst klasöründe models ve template bulunmalı.

EfficientNet varsayılan yolu: models/efficientnet_animals.pth. MODEL_PATH ortam değişkeni ile değiştirilebilir.

Kendi CNN'in şu anda pasif. Eklemek için app.py içindeki MODEL_CONFIGS sözlüğünde own_cnn bölümünün path alanını değiştir:

```python
"path": BASE_DIR / "models" / "own_cnn_animals.pth",
```

animal_model.py dosyan app.py ile aynı klasörde olsun. Gönderdiğin AnimalModel mimarisi için builder hazır; mimariyi değiştirirsen build_own_cnn fonksiyonunu da uyarla. Her model kendi eğitimindeki sınıf sırasıyla model_state_dict ve class_names içeren kayıt kullanmalı. Yalnız state_dict içeren kayıt yeterli değil.

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
}, "models/own_cnn_animals.pth")
```

EfficientNet weights.transforms() kullanır. Kendi CNN'in 128x128 boyut ve gönderdiğin mean/std değerlerini kullanır. Eğitim ayarların değişirse modelin transformunu da güncelle.

Sunucuyu yeniden başlat ve python app.py çalıştır. http://127.0.0.1:8000 adresini aç. İlk tahmin model yüklemesi nedeniyle daha yavaş olabilir. Modeller CPU'da önbelleğe alınır; her istekte seçilen model GPU'ya taşınır, işlem sonunda CPU'ya geri alınır. Böylece iki modelin ağırlıkları aynı anda GPU'da tutulmaz; model değiştirme ek aktarım süresi getirir.

GET /models model seçeneklerini döndürür. POST /predict multipart file ve model_id alanlarını alır. Kimlikler efficientnet ve own_cnn. Model dosyasının varlığı menüde kullanılabilirliği belirler; kayıt/mimari doğrulaması ilk tahminde yapılır. Geçersiz kayıt 503 yanıtı verir; ayrıntı terminalde görülür.

Python ve tarayıcı JavaScript sözdizimi kontrolü yapılmıştır. Gerçek model ağırlıkları sağlanmadığından uçtan uca tahmin doğrulanmamıştır.
