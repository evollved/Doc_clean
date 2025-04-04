```markdown
# Document Scanner & Processor

Проект для автоматического выравнивания, улучшения и преобразования сканов документов в PDF с удалением теней и коррекцией цветового баланса.

## 📌 Возможности

- **Автоматическое определение ориентации** текста (0°, 90°, 180°, 270°)
- **Ручная коррекция поворота** (интерактивный режим)
- **Точное выравнивание документа** по 4 точкам (авто/ручной режим)
- **Удаление теней** с адаптивным алгоритмом
- **Баланс белого** по среднему серому
- **Экспорт в PDF** с настройкой DPI

## 🖥️ Использование

1. Запустите программу:

./doc_clean.bin

2. Выберите изображения через диалоговое окно

3. Для каждого документа:
   - Откорректируйте поворот (клавиши A/D, Enter для подтверждения)
   - Отрегулируйте углы выравнивания (перетаскивайте точки)
   - Нажмите Enter для обработки

4. Результаты сохранятся:
   - Для одного файла: `[имя_файла]_processed.pdf`
   - Для нескольких: `processed_documents.pdf` в папке с исходниками

## 🛠 Технологии

- OpenCV (выравнивание, обработка изображений)
- Алгоритм Sauvola (бинаризация)
- Matplotlib (экспорт в PDF)
- Tkinter (GUI интерфейс)

## Примеры в файлах
1               - оригинальный файл
1_procecced     - скриншот обработанного файла
1_processed.pdf - получившийся обработанный файл

## 📜 Лицензия

MIT License.
