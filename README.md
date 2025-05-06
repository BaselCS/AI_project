# مقدمة

في حال كان هذا أول تشغيل لك للمشروع استخدم الأمر `pip install neat-python pygame matplotlib` لتحميل الحزم اللازمة

# بدء التدريب

في حال كنت تريد بدء فقط استخدم الأمر `python main.py`

# اختبار أفضل نسخة

في حال كنت تريد اختبار أفضل نسخة أستخدم الأمر `python test.py`

# تجربة اللعبة يدويًا

أذهب إلى ملف `main.py` و ألصق التالي بعد `if __name__ == '__main__':` :

```
if __name__ == '__main__':
    #أقرا readme.md
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, 'config.txt')

    SAVE_DIR = os.path.join(local_dir, 'dino_saves')
    OLD_DIR = os.path.join(local_dir, 'old_saves')


    #run(config_path)
     Game_loop(pygame.time.Clock(),is_AI=False)
```

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
