# مقدمة

في حال كان هذا أول تشغيل لك للمشروع استخدم الأمر `pip install neat-python pygame` لتحميل الحزم اللازمة

# بدء التدريب

في حال كنت تريد بدء فقط استخدم الأمر `python main.py`

# اختبار أفضل نسخة

في حال كنت تريد اختبار أفضل نسخة أستخدم الأمر `python test.py`

# تمثيل التدريب

في حال كنت تريد تمثيل التدريب الأخير اذهب إلى `graph.py` و ألصق التالي في بعد `if __name__ == "__main__":` :

```python
if __name__ == "__main__":
	 plot_best_fitness_per_generation()
```

في حال تدريب تمثيل كل التدريب السابقة أنسخ ألصق التالي :

```
if __name__ == "__main__":
    for dic in os.listdir(OLD_DIR):
        if os.path.isdir(os.path.join(OLD_DIR, dic)):
            print(f"Processing directory: {dic}")
            plot_best_fitness_per_generation(os.path.join(OLD_DIR, dic))
```
