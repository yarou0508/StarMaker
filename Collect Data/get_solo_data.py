# coding=utf-8
import time
from datetime import datetime, timedelta
import sys
from pyspark.sql import SparkSession

def init_spark(app_name):
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc

#def map_null_user(x):
    #if x.action=="click":
        #user_id = -1 if x.user_id is None else x.user_id
        #song_id = -1 if x.song_id is None else x.song_id
        #album_id = -1 if x.album_id is None else x.album_id
        #playlist_id = -1 if x.playlist_id is None else x.playlist_id
        #singer_id = -1 if x.singer_id is None else x.singer_id
        #return user_id, song_id, album_id, playlist_id, singer_id, x.action, x.search_keyword, x.search_type, x.page, x.timestamp, x.dt
    #else:
        #return x.user_id, x.song_id, x.album_id, x.playlist_id, x.singer_id, x.action, x.search_keyword, x.search_type, x.page, x.timestamp, x.dt

def map_null_user(x):
    user_id = -1 if x.user_id is None else x.user_id
    song_id = -1 if x.song_id is None else x.song_id
    album_id = -1 if x.album_id is None else x.album_id
    playlist_id = -1 if x.playlist_id is None else x.playlist_id
    singer_id = -1 if x.singer_id is None else x.singer_id
    return user_id, song_id, album_id, playlist_id, singer_id, x.action, x.search_keyword, x.search_type, x.timestamp, x.dt

def find_10000_user(x):
    user_id = []
    for line in x:
        if line[1] < 10000:
            user_id.append(str(line[0]))
    return user_id


if __name__ == "__main__":
    spark, sc = init_spark('solo_data_train')
    day = str(sys.argv[1])
    date = str(sys.argv[2])
    t = time.strptime(day, '%Y%m%d')
    y, m, d = t[0:3]
    top_user = sc.textFile("cosn://starmaker-research-1256122840/solo-listen-data/id-mapping/%s/uids" % day).map(lambda x:eval(x)).collect()
    user = find_10000_user(top_user)

    solo_data = []
    for i in range(7):
        dayi = (datetime(y, m, d) - timedelta(days=i)).strftime('%Y%m%d')
        SQL = '''
        SELECT user_id, song_id, album_id, playlist_id, singer_id, action, search_keyword, search_type, `timestamp`, dt 
        FROM solo_raw_data.event_fields_selected
        WHERE type == "search" AND dt=%s
        ''' % dayi
        solo_data.extend(spark.sql(SQL).rdd.map(map_null_user).collect())
    m = []
    i = 0
    for line in solo_data:
        if str(line[0]) in user:
            l1 = str(line[0]) + "@cc@" + str(line[1]) + "@cc@" + str(line[2]) + "@cc@" + str(line[3]) + "@cc@" + str(line[4]) + "@cc@" + str(line[5]) + "@cc@" + str(line[6].encode("utf-8")) + "@cc@" + str(line[7]) + "@cc@" + str(line[8]) + "@cc@" + str(line[9]) + "@cc@" + str(i)
            m.append(l1)
            i += 1

    solo_data_file = 'cosn://starmaker-research/yarou.xu/solo_data_rdd/train/%s'
    sc.parallelize(m).repartition(1).saveAsTextFile(solo_data_file % date)
    spark.stop()


#spark-submit get_solo_data.py 20180519 20180519
#spark-submit --queue C get_solo_data.py 20180618 0
