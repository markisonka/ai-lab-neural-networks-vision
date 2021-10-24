# Отчет по лабораторной работе 
## по курсу "Искусственый интеллект"

## Нейросетям для распознавания изображений


### Студенты: 

| ФИО             | Роль в проекте                                                | Оценка       |
|-----------------|---------------------------------------------------------------|--------------|
| Хисамутдинов Данил | Нейросеть, Подготовка датасета, Обучение нейросети            |              |
| Подоляка Елена | Подготовка датасета, Однослойная нейросеть, Подготовка отчета |              |
| Рыженко Иван    | Подготовка датасета,  Подготовка отчета                       |              |

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В.     |              |             |



## Тема работы

Классификация нейросетями (однослойной, многослойной, сверточной) набора данных, состоящего из букв Э, Ю, Я.

## Распределение работы в команде

Хисамутдинов Данил - обучение нейросети, общий шаблон датасета

Подоляка Елена - подготовка датасета, редактирование изображений, подготовка отчета

Рыженко Иван - подготовка датасета, редактирование изображений, подготовка отчета

## Подготовка данных

Исходные листки с рукописными символами:

![Э](https://user-images.githubusercontent.com/47860210/133614934-e20b65f2-19a2-404c-8316-bf3c18ba45d4.jpg) ![Ю](https://user-images.githubusercontent.com/47860210/133615167-eac927d4-a867-40ec-8ba1-29107517faa6.jpg) ![Я](https://user-images.githubusercontent.com/47860210/133615204-c35237c5-e03d-4b24-9dec-08c554126b9c.jpg)

![IMG_20210916_121635](https://user-images.githubusercontent.com/47860210/133615666-4b4a0383-d503-4c00-948f-66398f06181d.jpg) ![IMG_20210916_121736](https://user-images.githubusercontent.com/47860210/133615679-76b11117-0c59-4a5f-a630-adb6316e4974.jpg) ![IMG_20210916_121858](https://user-images.githubusercontent.com/47860210/133615683-833ef2fa-01e8-468e-b528-81ded147d0d2.jpg)

![II11](https://user-images.githubusercontent.com/47860210/133615452-d6c25d6a-8d55-4950-a737-644d4ed1dbcd.jpg) ![II22](https://user-images.githubusercontent.com/47860210/133615465-5126c79d-0a3f-4230-aa47-efec34945206.jpg) ![II33](https://user-images.githubusercontent.com/47860210/133615471-6f0527c1-dc4c-48ab-8f17-dc2be388bb54.jpg)



Как осуществлялась подготовка датасета? С какими сложностями пришлось столкнуться? Фрагменты кода для разрезания картинки на части...
Подготовка датасета осуществлялась с помощью библиотеки openCV. Размер изображений 320х320, выставлен с помощью split. Изображения делились на 10 квадратов. К этим выделенным квадратам были приписаны классы картинок. Данные были перемешаны и поделены на выборку 80/20.



## Загрузка данных
После подготовки данные были загружены:

    (train_images, train_labels), (test_images, test_labels) = data([["andrew_pos.jpg", "valera_pos.png", "vita_pos.png"],
                                                                 ["andrew_neut.jpg", "valera_neut.png", "vita_neut.png"],
                                                                 ["andrew_neg.jpg", "valera_neg.png", "vita_neg.png"]])
## Обучение нейросети

### Полносвязная однослойная сеть
Полносвязная сеть с внутренним слоем дает более высокие результаты, чем однослойная сеть. Полносвязный слой (Dense) и дополнительный (Flatten):

    self.model = keras.Sequential([
            layers.Flatten(),
            #layers.Dense(2048, activation='relu'),
            layers.Dense(3, input_shape=(32, 32, 3))
        ])
        
![image](https://user-images.githubusercontent.com/45311390/133831926-ff39d328-d15c-47c4-8e16-92ebd4113f4c.png)
        
### Полносвязная многослойная сеть
Применены полносвязные слои (Dense) и 1 выравнивающий (Flatten):

    self.model = keras.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(3)
        ])
        
![image](https://user-images.githubusercontent.com/45311390/133831958-56d308cb-3922-4695-8222-c46839601441.png)
        
График обучения нейросети выглядит достаточно хорошо, полученная точность для тестовой выборки = *******

Попробуем изменить число слоёв и нейронов в слоях полносвязной модели:

    self.model = keras.Sequential([
            layers.Flatten(),
            #layers.Dense(2048, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(3)
        ])
        
![image](https://user-images.githubusercontent.com/45311390/133832020-4b5f1eff-986d-4bd1-91ee-29eab2cf2f6e.png) ![image](https://user-images.githubusercontent.com/45311390/133832039-3b91ac34-80f9-40db-bcd3-84bd20ed8b22.png)


### Свёрточная сеть

    self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(3))
![image](https://user-images.githubusercontent.com/45311390/133832085-ef760ef5-6f03-44c6-93db-b4edf91f11a1.png) ![image](https://user-images.githubusercontent.com/45311390/133832098-2048727c-96bf-4f98-b27e-016fb1a33406.png)
![image](https://user-images.githubusercontent.com/45311390/133832118-08ea9f53-7d6f-479d-a12a-0ba6bfb1edc4.png)


## Выводы
По результатам тестов делаем вывод, что очевидно лучше свёрточная сеть. Она сеть учится быстрее и достигает лучших результатов. Время обучения в данной реализации приемлимо, благодаря правильно подобранным параметрам.

Так же требуется разнообразная выборка. Чем больше экземпляров, тем лучше. Кроме того, влияет качество написания символов.

Из сложностей в ходе выполнения лабораторной можно выделить высокий порог входа Tenserflow, большие затраты времени на подбор параметров и тестирование. Создание датасета, напротив, не вызвало особых проблем. 
