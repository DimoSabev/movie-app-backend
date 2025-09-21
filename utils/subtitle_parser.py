import re
import pysrt

def parse_srt(file_path, max_lines_per_chunk=5):
    subs = pysrt.open(file_path)
    scenes = []
    current_chunk = []
    start_time = None

    for i, sub in enumerate(subs):
        if not start_time:
            start_time = sub.start.to_time()

        current_chunk.append(sub.text)


        if len(current_chunk) >= max_lines_per_chunk or i == len(subs) - 1:
            scenes.append({
                "text": " ".join(current_chunk),
                "timestamp": str(start_time)
            })
            current_chunk = []
            start_time = None

    return scenes

def parse_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    parsed = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            timestamp_line = lines[1]
            text_lines = lines[2:]


            start_time_raw = timestamp_line.split('-->')[0].strip()
            start_time_clean = start_time_raw.split(',')[0]


            full_text = ' '.join(text_lines)


            clean_text = re.sub(r'<.*?>', '', full_text).strip()

            if clean_text:
                parsed.append({
                    "text": clean_text,
                    "timestamp": start_time_clean
                })

    return parsed
