from collections import Counter
import re
import string
import json
import os


def find_en_first(text):
    pattern = r'\((.*?)\)'
    match = re.search(pattern, text)
    if match:
        return text[:match.start()].strip(), match.group(1)
    else:
        return '', ''


def find_en(text):
    maxlen_flag = True
    if ')' in text:
        res = text[:text.find(')')].strip()
    else:
        res = text
        maxlen_flag = False
        # print
        print(f'!!!!!!!!!!max toekn is small!!!!!!!!!!')
    return maxlen_flag, res


def find_input(text):
    start = text.find('原文：')
    end = text.find('翻译结果：')
    content = text[start + 3:end].strip().split('\n')
    print(f'----content---{content}')
    return content


def is_not_mostly_english(text):
    english_chars = string.ascii_letters
    total_count = 0
    english_count = 0

    for char in text:
        if char.isalpha():
            total_count += 1
        if char in english_chars:
            english_count += 1

    if total_count == 0:
        return False

    return english_count / total_count < 0.3


def is_mostly_english(text):
    english_chars = string.ascii_letters
    total_count = 0
    english_count = 0

    for char in text:
        if char.isalpha():
            total_count += 1
        if char in english_chars:
            english_count += 1
    if total_count == 0:
        return False
    return english_count / total_count > 0.6 and english_count >= 1


def extract_pre_chinese_chars(s):
    match = re.search(r'[\u4e00-\u9fff]', s)
    if match:
        return s[:match.start()].strip()
    return s.strip()


def trim_non_chinese_chars(s):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    if not s:
        return s
    match = chinese_pattern.search(s)
    if match:
        first_chinese_idx = match.start()
    else:

        return ""
    s = s[first_chinese_idx:]
    match = chinese_pattern.search(s[::-1])
    if match:
        last_chinese_idx = len(s) - match.start()
    else:
        return ""
    s = s[:last_chinese_idx]

    return s


def remove_non_english_chars(s):
    return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', s)


def remove_non_english_chars_left(s):
    return re.sub(r'^[^a-zA-Z]+', '', s)


def remove_chinese_chars_and_brackets(s):
    s = re.sub(r'^[\u4e00-\u9fff]+', '', s)
    s = re.sub(r'[()（）]', '', s)
    return s


def remove_non_chinese_chars_and_brackets(s):
    s = re.sub(r'^[^\u4e00-\u9fff]+', '', s)
    s = re.sub(r'[()（）]', '', s)
    return s


def is_substring(str1, str2):
    return str1 in str2 or str2 in str1


def append_to_json_and_jsonl(filename_json, filename_jsonl, data):
    if not os.path.isfile(filename_json):
        with open(filename_json, 'w', encoding='utf-8') as f:
            json.dump([data], f, ensure_ascii=False, indent=4)
    else:
        with open(filename_json, 'r+', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.append(data)
                else:
                    existing_data = [existing_data, data]
            except json.JSONDecodeError:
                existing_data = [data]
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    with open(filename_jsonl, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
