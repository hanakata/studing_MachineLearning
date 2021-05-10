def parse_arguments():
    """
    コマンドライン用のシンプルな引数パーサ
    :return 編集されるテキスト
    """
    parser = argparse.ArgumentParser(
        description = "編集するテキストを受け取る"
    )
    parser.add_argument(
        'text',
        metaver='input text',
        type = str
    )
    args = parser.parse_args()
    return args.text

def clean_input(text):
    return str(text.encode().decode('ascii', errors='ignore'))

def preprocess_input(text):
    sentences = nltk.sent_tokkenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return token

def get_suggestions(sentence_list):
    told_said_usage = sum(
        (count_word_usage(tokens,["told","said"]) for tokens in sentence_list)
    )
    but_and_usage = sum(
        (count_word_usage(tokens,["but","and"]) for tokens in sentence_list)
    )
    wh_adverbs_usage = sum(
        (
            count_word_usage(
                tokens,
                [
                    "when",
                    "where",
                    "why",
                    "whence",
                    "whereby",
                    "wherein",
                    "whereupon"
                ],
            )
            for tokens in sentence_list
        )
    )
    result_str = ""
    adverb_usage = "Adverb usage: %s told/said, %s but/and, %s wh adverbs" %(
        told_said_usage,
        but_and_usage,
        wh_adverbs_usage,
    )
    result_str += adverb_usage
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_faction(sentence_list)

    word_stats = "Average word length %.2f, fraction of unique words %.2f" % (
        average_word_length,
        unique_words_fraction,
    )
    result_str += "<br/>"
    result_str += word_stats

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)

    syllable_counts = "%d syllables, %d words, %d sentences" % (
        number_of_syllables,
        number_of_words,
        number_of_sentences,
    )
    result_str += "<br/>"
    result_str += syllable_counts

    flesch_score = compute_flesch_reading_ease(
        number_of_syllables,number_of_words,number_of_sentences
    )
    flesch = "%d syllables, %.2f flesch score: %s" %(
        number_of_syllables,
        flesch_score,
        get_reading_level_from_flesch(flesch_score),
    )
    result_str += "<br/>"
    result_str += flesch

    return result_str

input_text = parse_arguments()
processed = clean_input(input_text)
tokenized_sentences = preprocess_input(processed)
suggestions = get_suggestions(tokenized_sentences)