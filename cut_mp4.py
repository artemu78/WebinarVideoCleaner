import json
import os
from datetime import timedelta
# Современный импорт для MoviePy 3.0+
from moviepy import VideoFileClip, concatenate_videoclips

def time_to_seconds(t_str):
    """Превращает HH:MM:SS или MM:SS в секунды."""
    parts = list(map(int, t_str.split(':')))
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2: return parts[0] * 60 + parts[1]
    return 0

def format_seconds(seconds):
    """Форматирует секунды в HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def process_video(video_path, json_path):
    """
    Обрабатывает видео, вырезая указанные фрагменты.
    
    Args:
        video_path: Путь к входному видео файлу
        json_path: Путь к JSON файлу с данными о вырезах
        
    Returns:
        str: Путь к выходному файлу или None в случае ошибки
    """
    # Save output file in the same directory as the input video
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_basename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_basename)
    output_path = os.path.join(video_dir, f"cleaned_{video_name}{video_ext}")

    if not os.path.exists(video_path) or not os.path.exists(json_path):
        print("Ошибка: Файлы не найдены. Проверьте пути.")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            cuts_data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения JSON: {e}")
        return None

    # Validate that cuts_data is a list
    if not isinstance(cuts_data, list):
        print(f"Ошибка: JSON должен содержать список, получен {type(cuts_data)}")
        return None
    
    # Check if list is empty
    if len(cuts_data) == 0:
        print("Предупреждение: Список вырезов пуст. Видео не будет изменено.")
        # Return the original video path or create a copy
        return None

    video = VideoFileClip(video_path)
    initial_length = video.duration
    
    cut_intervals = []
    total_cut_duration = 0
    for c in cuts_data:
        # Validate that each cut item is a dict with 'start' and 'end' keys
        if not isinstance(c, dict):
            print(f"Предупреждение: Пропущен неверный элемент выреза: {c}")
            continue
        
        if 'start' not in c or 'end' not in c:
            print(f"Предупреждение: Пропущен элемент выреза без 'start' или 'end': {c}")
            continue
        
        try:
            start = time_to_seconds(c['start'])
            end = time_to_seconds(c['end'])
            
            # Validate that start < end
            if start >= end:
                print(f"Предупреждение: Пропущен неверный диапазон (start >= end): {c['start']} - {c['end']}")
                continue
            
            # Ограничиваем конец выреза длительностью видео
            end = min(end, initial_length)
            
            # Skip if start is beyond video length
            if start >= initial_length:
                print(f"Предупреждение: Пропущен диапазон за пределами видео: {c['start']} - {c['end']}")
                continue
            
            # Ensure start is not negative
            start = max(0, start)
            
            cut_intervals.append((start, end))
            total_cut_duration += (end - start)
        except (ValueError, KeyError) as e:
            print(f"Предупреждение: Пропущен элемент выреза из-за ошибки: {c}, ошибка: {e}")
            continue
    
    cut_intervals.sort()

    # Собираем сегменты, которые ОСТАВЛЯЕМ
    keep_segments = []
    last_end = 0
    for start, end in cut_intervals:
        if start > last_end:
            # В MoviePy 3.0 используем subclipped вместо subclip
            keep_segments.append(video.subclipped(last_end, start))
        last_end = end
    
    if last_end < initial_length:
        keep_segments.append(video.subclipped(last_end, initial_length))

    if keep_segments:
        final_video = concatenate_videoclips(keep_segments)
        final_length = final_video.duration
        
        print("\n" + "="*40)
        print("ОТЧЕТ О ДЛИТЕЛЬНОСТИ:")
        print(f"Исходное видео:   {format_seconds(initial_length)}")
        print(f"Вырезано (всего): {format_seconds(total_cut_duration)}")
        print(f"Итоговое видео:   {format_seconds(final_length)}")
        print("="*40 + "\n")

        # Сохранение
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        final_video.close()
    else:
        print("Ошибка: После удаления фрагментов ничего не осталось.")
        video.close()
        return None

    video.close()
    print(f"\nГотово! Файл сохранен: {output_path}")
    return output_path

if __name__ == "__main__":
    print("--- MoviePy 3.0 Video Cutter ---")
    video_path = input("Введите название видео: ").strip()
    json_path = input("Введите название файла с вырезами: ").strip()
    output_path = process_video(video_path, json_path)
    if output_path:
        print(f"Output file: {output_path}")