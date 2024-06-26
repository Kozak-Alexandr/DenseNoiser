# Обзор уже существующих методов
Я буду рассматривать только *realtime* модели для подавления шума


## Собственно, baseline [RNNoise](https://arxiv.org/pdf/1709.08243.pdf)
То, что я его выбрал в качестве *baseline* говорит уже само за себя. 
- Низкая сложность
- Реал-тайм
- Работает на CPU
- Может принимать VAD как вход для большей точности
- Барк-спектр 
- Питч-фильтр, чтобы чистить между гармониками

Классика.
![](/Pics/RNNoise.jpg)
![](/Pics/RNNoise1.jpg)

## [DeepFilterNet 2](https://arxiv.org/pdf/2205.05474.pdf)
- Реалтайм на CPU (даже на Raspberry Pi)
- Верхнюю часть спектра (выше 5 кГц) обрабатывает сетка попроще, нижнюю часть – посложнее
- Верхняя часть – ERB фичи, нижняя – STFT 
- Одномерные свертки по частотам (есть двумерная свертка в энкодере), GRU – по времени
  
Двухэтапный процесс шумоподавления с использованием глубокой фильтрующей сетки: Весь первый этап работает в сжатой области ERB, которая служит цели снижения вычислительной сложности при моделировании слухового восприятия человеческого уха. Таким образом, целью первого этапа является улучшение огибающей речи с учетом ее грубого частотного разрешения. Второй этап работает в сложной области, используя глубокую фильтрацию, и пытается восстановить периодичность речи.
![](/Pics/DeepFilter.jpg)
![](/Pics/DeepFilter1.jpg)
![](/Pics/DeepFilter2.jpg.png)

### И тут мы начинаем понимать, что свёртки(а еще GRU) это круто

## [Conv-TasNet](https://arxiv.org/pdf/1809.07454.pdf)
Данная технология базируется на методе масок, его основа - сверточные нейронные сети Conv-TasNet
Предшественник этой архитектуры – TasNet. Архитектура TasNet состоит из сверточных энкодера и декодера с некоторыми особенностями: 
выход энкодера ограничен значениями от нуля до бесконечности [0, ∞);
линейный декодер конвертирует выход энкодера в акустическую волну;
подобно многим методам-предшественникам на основе спектрограмм, на последнем этапе система аппроксимирует взвешивающую функцию (в данном случае LSTM) для каждого момента времени.
Conv-TasNet – модификация алгоритма TasNet, которая использует в качестве взвешивающей функции сверточные слои с расширением (dilation). Это модификация была сделана после того, как свертки с расширением показали себя эффективным алгоритмом при одновременном анализе и генерации данных переменной длины, в частности, для синтеза в таких решениях, как WaveNet.
Подход для разделения аудио/шумоподавления Conv-TasNet состоит из 3-х компонентов 
![](/Pics/ConvTasNet.png)
Основной компонент в схеме на картинке – этап разделения. Этот этап решает проблему приближенного исчисления источников, смесь которых мы рассматриваем в качестве «грязных» примеров. Формально предположение о «смешанности» нашего сигнала можно выразить следующим образом:
**x(t) = i=tCSi(t)**
Где x(t) - смесь в определенный момент времени, С - количество источников, несущих вклад в смесь, S1(t)...Sc(t) - источники в определенный момент времени.
Задача алгоритма машинного обучения – определить источники s1(t), … , sc(t), зная заранее количество источников C и смесь x(t).
Разделение в алгоритме происходит не сразу, а только после извлечения признаков из сигнала с помощью «1D блоков» 
![](/Pics/ConvTasNet1.png)
![](/Pics/ConvTasNet2.png)

# Мой любимый [DenseNet](https://arxiv.org/abs/1404.1869)
Архитектура DenseNet, или Densely Connected Convolutional Networks, представляет собой глубокую сверточную нейронную сеть с резидуалами не через слой, а со всех слоев во все. Основная идея DenseNet заключается в том, чтобы сделать связи между слоями еще более плотными по сравнению с другими архитектурами, такими как ResNet, что позволяет эффективно передавать информацию через все уровни сети.
![](/Pics/DANCEE.png)

## Вот eго ключевые особенности:

### Dense Blocks
Основной строительный блок архитектуры DenseNet. Внутри каждого блока каждый слой принимает на вход выходы всех предыдущих слоев и передает свой выход следующему слою. Это создает крайне плотные связи между слоями.
### Transition Blocks 
Используются для уменьшения размера карт признаков между блоками плотной связности. Они содержат сверточные слои и слой пулинга, что помогает уменьшить количество параметров и вычислений в сети.
### Global Pooling 
В конце архитектуры обычно следует глобальный пулинг, который усредняет признаки по всему пространству, чтобы получить окончательный вектор признаков для классификации.

## Преимущества


### Сокращение проблемы затухания градиента:
Благодаря коротким путям обратного распространения ошибки, обеспечиваемым плотными связями, DenseNet помогает справиться с проблемой затухания градиента в глубоких нейронных сетях.
### Эффективное использование параметров: 
Поскольку каждый слой получает входные данные от всех предыдущих слоев, нет необходимости в большом количестве параметров для передачи информации по слоям.
### Высокая точность: 
DenseNet демонстрирует хорошие результаты на различных наборах данных для классификации изображений и сегментации. Ну а здесь я попытался использовать его для *noise supression* 

### However...Есть и недостатки...


### Вычислительная сложность 
Плотные связи между слоями могут привести к увеличению вычислительной сложности и требованиям к памяти при обучении модели(Об этом позже).
### Чувствительность к гиперпараметрам 
Подбор гиперпараметров, таких как количество блоков и количество фильтров в блоке, может потребовать больших вычислительных ресурсов.


# Моя реализация DenseNet Для подавления шума

![Архитектура](/model_densenet.png)

Код для смешивания аудио [NoisyDataMaker.py](https://github.com/Kozak-Alexandr/DenseNoiser/blob/main/NoisyDataMaker.py)
Я смешивал 32bit 16000hz аудио между собой, так что для препроцессинга особой нужды не было, плюс все аудио были примерно равной громкости. Если бы было больше вреени, то смешивал бы разные уровни шума, или добавил бы шумов из других датасетов.
Код для тренировки модели вместе с её архитектурой [ArchitectureAndTraining.py](https://github.com/Kozak-Alexandr/DenseNoiser/blob/main/ArchitectureAndTraining.py)

Чтобы убедиться в том, что модель работает, можно запустить [demonstration.py](https://github.com/Kozak-Alexandr/DenseNoiser/blob/main/demonstration.py) закинув туда путь до папки с интересующими файлами

Для обучения модели я использовал [zenodo](https://zenodo.org/records/1227121) и [openslr](https://www.openslr.org/)

Обученные модели и сгенерированные аудио(в том числе и предсказания на тестовом датасете) можно найти [на диске](https://disk.yandex.ru/d/dglhBJF_MW0hrQ) 
Некоторые аудио из тестового датасета очень забавно *enhancятся* так, что речь становится немного жутковатой. Дело в том, что мне не хватило терпения обучить модель до приемлемых результатов. Однако пайплайн рабочий и всё должно работать

Как я и говорил выше, DenseNet довольно тяжело обучать, а на старом ноуте со старым TensorFlow ещё сложнее. Оказалось, что видеопамяти моей видеокарты мало для обработки приемлемых батчей, так что обучал я на CPU + RAM. Однако можно было для обучения использовать меньшее количество данных (например рандомно выбирать какую-то часть или файлы для тренировки). Также можно было упростить архитектуру, добавив побольше *bottleneck* слоёв, уменьшить количество слоёв для каждого *dense block*, также можно было бы уменьшить *growth rate*. В итоге большинство времени было потрачено на разборки с железом. Позже уговорил знакомого отдать мне RTX3060 на полдня, но обучиться успело только 4 эпохи, взглянуть на эту модель можно тут [DanseNoise_RTX](https://disk.yandex.ru/d/dglhBJF_MW0hrQ).
Вывод здесь один: мощное железо ой как не помешало бы. С RTX360 получается обучать больше одной эпохи в час, а это уже приятно (за 4 эпохи на RTX mos_pred = 1.87). Если бы у меня был доступ к каким-нибудь теслам, или другим мощным gpu, то от предсказаний можно было бы ожидать лучшего. Также стоило бы переписать код на Torch, ибо есть мнение, что он всё-такие новее, оптимизированнее и с видеокартой проблем быть не должно, как это было в TF.

## Сравнение с baseline на тестовом датасете

Табличка со средними NISQA [Predictions](https://github.com/Kozak-Alexandr/DenseNoiser/blob/main/Predictions.py)
| Model      | mos_pred | noi_pred | dis_pred | col_pred | loud_pred |
|------------|----------|----------|----------|----------|-----------|
| NISQA      | 2.45     | 1.85     | 4.12     | 3.53     | 3.32      |
| RNNoise    | 1.48     | 3.24     | 2.77     | 2.51     | 3.47      |
| DenseNoiser| 1.94     | 1.69     | 3.72     | 2.86     | 2.89      |









