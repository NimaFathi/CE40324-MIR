def jaccard_similarity(word1, word2):
    # intersection = len(list(set(word1).intersection(word2)))
    # union = (len(word1) + len(word2)) - intersection
    # return float(intersection) / union

    bigrams1 = [word1[i:i + 2] for i in range(len(word1) - 1)]
    bigrams2 = [word2[i:i + 2] for i in range(len(word2) - 1)]
    intersection = len(list(set(bigrams1).intersection(set(bigrams2))))
    union = len(set(bigrams1)) + len(set(bigrams2)) - intersection
    return float(intersection) / union


def similar_words_j(dictionary, in_word, threshold):
    jaccard_list = []
    for dest_word in dictionary:
        jaccard_list.append((dest_word, jaccard_similarity(in_word, dest_word)))
    jaccard_list = sorted(jaccard_list, key=lambda x: x[1], reverse=True)
    similars = []
    for item in jaccard_list:
        if item[1] >= threshold:
            similars.append(item[0])
        else:
            break
    return similars


def levenshtein_distance(word1, word2):
    if len(word1) > len(word2):
        word1, word2 = word2, word1

    distances = range(len(word1) + 1)
    for i2, c2 in enumerate(word2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(word1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def similar_words_l(dictionary, in_word):
    distance_list = []
    for dest_word in dictionary:
        distance_list.append((dest_word, levenshtein_distance(dest_word, in_word)))
    distance_list = sorted(distance_list, key=lambda x: x[1])
    min_distance = distance_list[0][1]
    similars = []
    for dist in distance_list:
        if dist[1] <= min_distance:
            similars.append(dist[0])
        else:
            break
    return similars


def correct_query(q, dictionary):
    modified_query = []
    # threshold = 0.5
    for word in q:
        if word in dictionary:
            modified_query[len(modified_query):] = [word]
        else:
            result_l = similar_words_l(dictionary, word)
            modified_query.append(result_l)
            # result_j = similar_words_j(dictionary, word, threshold)
            # print(word, ":\n\n", "Jaccard: ",  result_j, "\nLevenshtein: ", result_l)

    return modified_query
