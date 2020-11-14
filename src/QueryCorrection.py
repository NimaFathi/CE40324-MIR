def jaccard_similarity(word1, word2):
    # intersection = len(list(set(word1).intersection(word2)))
    # union = (len(word1) + len(word2)) - intersection
    # return float(intersection) / union

    bigrams1 = [word1[i:i + 2] for i in range(len(word1) - 1)]
    bigrams2 = [word2[i:i + 2] for i in range(len(word2) - 1)]
    intersection = len(list(set(bigrams1).intersection(set(bigrams2))))
    union = len(set(bigrams1)) + len(set(bigrams2)) - intersection
    return float(intersection) / union


def similar_words_j(dictionary, in_word):
    jaccard_list = []
    for dest_word in dictionary:
        jaccard_list.append((dest_word, jaccard_similarity(in_word, dest_word)))
    jaccard_list = sorted(jaccard_list, key=lambda x: x[1], reverse=True)
    similars = []
    size = len(jaccard_list)
    if size > 10:
        size = 10
    for i in range(size):
        similars.append(jaccard_list[i][0])
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
    size = len(distance_list)
    if size > 10:
        size = 10
    min_distance = distance_list[size - 1][1]
    similars = []
    for dist in distance_list:
        if dist[1] <= min_distance:
            similars.append(dist[0])
        else:
            break
    return similars


def correct_query(q, dictionary):
    modified_query = []
    for word in q.split():
        if word in dictionary:
            modified_query[len(modified_query):] = [word]
        else:
            result_j = similar_words_j(dictionary, word)
            if len(result_j) == 0:
                result_l = similar_words_l(dictionary, word)
            else:
                result_l = similar_words_l(result_j, word)
            modified_query.append(result_l[0])
    return modified_query
