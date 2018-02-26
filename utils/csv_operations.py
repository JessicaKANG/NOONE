import time
import csv


def write_submission(name, tg_playlist, recs):
    """
    Write the recommendation result into .csv file for submission

    :param: name: name of the algorithm, for constructing submission name
    :param: tg_playlist: a list of target playlists
    :param: recs: a dictionary of recommendations
        'pl_id': ['tr_1', 'tr_2', 'tr_3', ...] for each playlist in tg_playlist
    :return:
    """

    # construct output file name
    ISOTIMEFORMAT = '%m%d%H%M'
    output = "submissions/" + name + '_' + str(time.strftime(ISOTIMEFORMAT)) + '.csv'

    with open(output, mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for pl_id in tg_playlist:
            track_ids = ''
            for tr_id in recs[pl_id]:
                track_ids = track_ids + tr_id + ' '
            writer.writerow({'playlist_id': pl_id,
                             'track_ids': track_ids[:-1]})
    print("[csv_writer]: Submission " + output + " is ready.")


def load_column(path, key):
    """
    :param path: path of the file to be loaded
    :param key: key of the column
    :return: a list of elements in this column
    """
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        column = [row[key] for row in reader]
    return column


def load_column_float(path, key):
    """
    :param path: path of the file to be loaded
    :param key: key of the column
    :return: a list of float elements in this column
    """
    column = []
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            if row[key] != '':
                column.append(float(row[key]))
    return column


def load_tracks(path):
    mapper_id_idx = {}
    mapper_idx_id = {}
    mapper_id_id_tr_attr = {'album': {}, 'artist_id': {}, 'tags': {}, 'playcount': {}, 'duration': {}}
    mapper_attr = {'album': {}, 'artist_id': {}, 'tags': {}, 'playcount': {}, 'duration': {}}
    mapper_attr_index = {'album': {}, 'artist_id': {}, 'tags': {}, 'playcount': {}, 'duration': {}}
    # for computing tag-title correlation
    mapper_id_tags = {}
    set_attr = {'album': set(), 'artist_id': set(), 'tags': set(), 'playcount': set(), 'duration': set()}
    # a dict of dict contains corpus for each attribute, used for computing tf-idf for all attr
    corpus_tr = {'album': {}, 'artist_id': {}, 'tags': {}, 'playcount': {}, 'duration': {}}
    index = 0
    index_attr = {'album': 0, 'artist_id': 0, 'tags': 0, 'playcount': 0, 'duration': 0}

    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            mapper_id_idx[row['track_id']] = index
            mapper_idx_id[index] = row['track_id']
            index += 1

            # album
            albums = parse_list(row['album'])
            if len(albums):
                for album in albums:
                    if album not in set_attr['album']:
                        set_attr['album'].add(album)
                        mapper_attr['album'][album] = index_attr['album']
                        mapper_attr_index['album'][index_attr['album']] = album
                        index_attr['album'] += 1
                    mapper_id_id_tr_attr['album'][row['track_id']] = album
                    count = corpus_tr['album'].setdefault(album, 0)
                    corpus_tr['album'][album] = count + 1
            else:
                mapper_id_id_tr_attr['album'][row['track_id']] = -1

            # tags
            tags = parse_list(row['tags'])
            if len(tags):
                # unique tags of each track
                tagset_local = set()
                for tag in tags:
                    tagset_local.add(tag)
                    if tag not in set_attr['tags']:
                        set_attr['tags'].add(tag)
                        mapper_attr['tags'][tag] = index_attr['tags']
                        mapper_attr_index['tags'][index_attr['tags']] = tag
                        index_attr['tags'] += 1
                for tag in tagset_local:
                    count = corpus_tr['tags'].setdefault(tag, 0)
                    corpus_tr['tags'][tag] = count + 1
                # buile track_id_tags_set mapper for computing tag-title correlation
                mapper_id_tags[row['track_id']] = tagset_local
                mapper_id_id_tr_attr['tags'][row['track_id']] = tagset_local
            else:
                mapper_id_tags[row['track_id']] = set()
                mapper_id_id_tr_attr['tags'][row['track_id']] = set()

            # artist_id
            artist_id = row['artist_id']
            if artist_id not in set_attr['artist_id']:
                set_attr['artist_id'].add(artist_id)
                mapper_attr['artist_id'][artist_id] = index_attr['artist_id']
                mapper_attr_index['artist_id'][index_attr['artist_id']] = artist_id
                index_attr['artist_id'] += 1
            mapper_id_id_tr_attr['artist_id'][row['track_id']] = artist_id
            count = corpus_tr['artist_id'].setdefault(artist_id, 0)
            corpus_tr['artist_id'][artist_id] = count + 1

            # duration
            duration = float(row['duration'])
            if duration != -1:
                if duration not in set_attr['duration']:
                    set_attr['duration'].add(duration)
                    mapper_attr['duration'][duration] = index_attr['duration']
                    mapper_attr_index['duration'][index_attr['duration']] = duration
                    index_attr['duration'] += 1
                mapper_id_id_tr_attr['duration'][row['track_id']] = duration
                count = corpus_tr['duration'].setdefault(duration, 0)
                corpus_tr['duration'][duration] = count + 1
            else:
                mapper_id_id_tr_attr['duration'][row['track_id']] = -1

            # playcount
            if row['playcount'] is not None and row['playcount'] != '':
                playcount = float(row['playcount'])
                if playcount not in set_attr['playcount']:
                    set_attr['playcount'].add(playcount)
                    mapper_attr['playcount'][playcount] = index_attr['playcount']
                    mapper_attr_index['playcount'][index_attr['playcount']] = playcount
                    index_attr['playcount'] += 1
                mapper_id_id_tr_attr['playcount'][row['track_id']] = playcount
                count = corpus_tr['playcount'].setdefault(playcount, 0)
                corpus_tr['playcount'][playcount] = count + 1
            else:
                mapper_id_id_tr_attr['playcount'][row['track_id']] = -1
    return mapper_id_id_tr_attr, mapper_id_idx, mapper_idx_id, mapper_attr, mapper_attr_index, index, index_attr, mapper_id_tags, corpus_tr


def load_playlists(path):
    mapper_id_index = {}
    mapper_index_id = {}
    mapper_index_id_ow = {}
    mapper_id_index_pl_ow = {}
    mapper_attr = {'owner': {}, 'title': {}, 'numtracks': {}, 'duration': {}, 'created_at': {}}
    mapper_id_titles = {}
    set_attr = {'owner': set(), 'title': set(), 'numtracks': set(), 'duration': set(), 'created_at': set()}
    # a dict {'title': num of playlists which has this title} used for computing tf-idf
    corpus = {}
    index = 0
    index_attr = {'owner': 0, 'title': 0, 'numtracks': 0, 'duration': 0, 'created_at': 0}

    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            mapper_id_index[row['playlist_id']] = index
            mapper_index_id[index] = row['playlist_id']
            index += 1

            # owner
            owner = row['owner']
            if owner not in set_attr['owner']:
                set_attr['owner'].add(owner)
                mapper_attr['owner'][owner] = index_attr['owner']
                mapper_index_id_ow[index_attr['owner']] = owner
                index_attr['owner'] += 1
            mapper_id_index_pl_ow[row['playlist_id']] = mapper_attr['owner'][owner]

            # title
            titles = parse_list(row['title'])
            if len(titles):
                # unique titles of each track
                titleset_local = set()
                for title in titles:
                    titleset_local.add(title)
                    if title not in set_attr['title']:
                        set_attr['title'].add(title)
                        mapper_attr['title'][title] = index_attr['title']
                        index_attr['title'] += 1
                for title in titleset_local:
                    count = corpus.setdefault(title, 0)
                    corpus[title] = count + 1
                mapper_id_titles[row['playlist_id']] = titleset_local
            else:
                mapper_id_titles[row['playlist_id']] = set()

            # created_at
            created_at = row['created_at']
            if created_at not in set_attr['created_at']:
                set_attr['created_at'].add(created_at)
                mapper_attr['created_at'][created_at] = index_attr['created_at']
                index_attr['created_at'] += 1

            # duration
            duration = float(row['duration'])
            if duration not in set_attr['duration']:
                set_attr['duration'].add(duration)
                mapper_attr['duration'][duration] = index_attr['duration']
                index_attr['duration'] += 1

            # numtracks
            numtracks = float(row['numtracks'])
            if numtracks not in set_attr['numtracks']:
                set_attr['numtracks'].add(numtracks)
                mapper_attr['numtracks'][numtracks] = index_attr['numtracks']
                index_attr['numtracks'] += 1

    return mapper_id_index, mapper_index_id, mapper_index_id_ow, mapper_id_index_pl_ow, mapper_attr, index, \
           index_attr, mapper_id_titles, corpus


def parse_list(str_list):
    """
    Parse the list in csv file into a real list
    :param str_list: a string list in form '[tag1, tag2, ...]'
    :return: a real list
    """
    str_cleaned = str_list.replace('[', '')
    str_cleaned = str_cleaned.replace(']', '')
    if str_cleaned == 'None' or str_cleaned == '':
        real_list = []
    else:
        real_list = str_cleaned.split(', ')
    return real_list


def load_interaction(path):
    """
    :param path: file path
    :return: a dict
    """
    interaction = {}
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            interaction.setdefault(row['playlist_id'], [])
            interaction[row['playlist_id']].append(row['track_id'])
    return interaction


