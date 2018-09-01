import time
from datetime import datetime, timedelta
import sys
from pyspark.sql import SparkSession
import numpy as np

def init_spark(app_name):
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def deal_null(x):
    if x is None:
        x = "0@cc@0@cc@0@cc@0@cc@0"
    else:
        x = x
    return x

def name_null(x):
    if x is None:
        x = "0@cc@0@cc@0"
    else:
        x = x
    return x

def take_log(x):
    return np.log(x+1)

if __name__ == "__main__":
    spark, sc = init_spark('get_feature')
    day = str(sys.argv[1])
    date = str(sys.argv[2])
    t = time.strptime(day, '%Y%m%d')
    y, m, d = t[0:3]

    # Get song, singer, album 'play', 'like', 'download', 'share', 'comment' number
    song_click = None
    singer_click = None
    album_click = None
    playlist_click = None
    for i in range(7):
        dayi = (datetime(y, m, d) - timedelta(days=i)).strftime('%Y%m%d')
        SQL_song = '''
        SELECT song_id, play_num, like_num, download_num, share_num, comment_num, dt 
        FROM solo_emr.song_stats
        WHERE dt=%s
        ''' % dayi
        SQL_singer = '''
        SELECT singer_id, play_num, like_num, download_num, share_num, comment_num, dt 
        FROM solo_emr.singer_stats
        WHERE dt=%s
        ''' % dayi
        SQL_album= '''
        SELECT album_id, play_num, like_num, download_num, share_num, comment_num, dt 
        FROM solo_emr.album_stats
        WHERE dt=%s
        ''' % dayi
        SQL_playlist = '''
        SELECT playlist_id, play_num, like_num, download_num, share_num, comment_num, dt 
        FROM solo_emr.playlist
        WHERE dt=%s
        ''' % dayi
        if song_click is None or singer_click is None or album_click is None or playlist_click is None:
            song_click = spark.sql(SQL_song).rdd
            singer_click = spark.sql(SQL_singer).rdd
            album_click = spark.sql(SQL_album).rdd
            playlist_click = spark.sql(SQL_playlist).rdd
        else:
            song_click.union(spark.sql(SQL_song).rdd)
            singer_click.union(spark.sql(SQL_singer).rdd)
            album_click.union(spark.sql(SQL_album).rdd)
            playlist_click.union(spark.sql(SQL_playlist).rdd)

    # Match name for song, singer, album and playlist
    raw_data = sc.textFile("cosn://starmaker-research-1256122840/yarou.xu/solo_data_rdd/train/test.txt").map(
        lambda x: x.split("@cc@")).map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), str( x[5]),str( x[6].encode("utf-8")), str( x[7]), str( x[8]), str( x[9]), int( x[10]), int( x[11]), int(x[12]))).cache()

    SQL_get_songname = '''
    SELECT song_id, name, album_name, singer_name1
    FROM solo_emr.s_song
    WHERE dt=%s
    ''' % day

    SQL_get_singername = '''
    SELECT singer_id, name 
    FROM solo_emr.s_singer
    WHERE dt=%s
    ''' % day

    SQL_get_albumname = '''
    SELECT album_id, name
    FROM solo_emr.s_album
    WHERE dt=%s
    ''' % day

    SQL_get_playlistname = '''
    SELECT playlist_id, name
    FROM solo_emr.playlist
    WHERE dt=%s
    ''' % day

    song_name = spark.sql(SQL_get_songname).rdd.map(lambda x: (int(x[0]), (str(x[1].encode("utf-8"))+ "@cc@" + str(x[2].encode("utf-8")) + "@cc@" + str(x[3].encode("utf-8")))))
    singer_name = spark.sql(SQL_get_singername).rdd.map(lambda x: (int(x[0]), str(x[1].encode("utf-8"))))
    album_name = spark.sql(SQL_get_albumname).rdd.map(lambda x: (int(x[0]), str(x[1].encode("utf-8"))))
    playlist_name = spark.sql(SQL_get_playlistname).rdd.map(lambda x: (int(x[0]), str(x[1].encode("utf-8"))))

    song_click1 = song_click.map(lambda (a, b, c, d, e, f, g): (
        str(a) + '@cc@' + str(g), (str(take_log(b)) + "@cc@" + str(take_log(c)) + "@cc@" + str(take_log(d)) + "@cc@" + str(take_log(e)) + "@cc@" + str(take_log(f)))))
    #singer_click1 = singer_click.map(lambda (a, b, c, d, e, f, g): (
        #str(a) + '@cc@' + str(g), (str(take_log(b)) + "@cc@" + str(take_log(c)) + "@cc@" + str(take_log(d)) + "@cc@" + str(take_log(e)) + "@cc@" + str(take_log(f)))))
    #album_click1 = album_click.map(lambda (a, b, c, d, e, f, g): (
        #str(a) + '@cc@' + str(g), (str(take_log(b)) + "@cc@" + str(take_log(c)) + "@cc@" + str(take_log(d)) + "@cc@" + str(take_log(e)) + "@cc@" + str(take_log(f)))))
    #playlist_click1 = playlist_click.map(lambda (a, b, c, d, e, f, g): (
        #str(a) + '@cc@' + str(g), (str(take_log(b)) + "@cc@" + str(take_log(c)) + "@cc@" + str(take_log(d)) + "@cc@" + str(take_log(e)) + "@cc@" + str(take_log(f)))))

    match_song = raw_data.filter(lambda x: x[7]=="song").map(lambda (a, b, c, d, e, f, g, h, i, j, k, l, m): (b, (a, f, g, h, i, j, k, l, m)))\
        .leftOuterJoin(song_name).map(lambda (b,((a,f,g,h,i,j,k,l,m), n)): (str(b) + "@cc@" + str(j), (a,b,f,g,h,i,j,k,l,m,n))).leftOuterJoin(song_click1)\
        .map(lambda (y, ((a, b, f, g, h, i, j, k, l, m, n), r)): (str(a)+ "@cc@" + str(b)+ "@cc@" + str(f)+ "@cc@" + str(g)+ "@cc@" + str(h)+ "@cc@" + str(i)+ "@cc@" + str(j)+ "@cc@" + str(k)+ "@cc@" + str(l)+ "@cc@" + str(m)+ "@cc@" + str(name_null(n)) + "@cc@" + str(deal_null(r))))

    #match_album = raw_data.filter(lambda x: x[7] == "album").map(lambda (a, b, c, d, e, f, g, h, i, j, k, l, m): (c, (a, f, g, h, i, j, k, l, m))) \
        #.leftOuterJoin(album_name).map(lambda (c, ((a, f, g, h, i, j, k, l, m), n)): (str(c) + "@cc@" + str(j),(a, c, f, g, h, i, j, k, l, m, n))).leftOuterJoin(album_click1) \
        #.map(lambda (y, ((a, c, f, g, h, i, j, k, l, m, n), r)): (str(a) + "@cc@" + str(c) + "@cc@" + str(f) + "@cc@" + str(g) + "@cc@" + str(h) + "@cc@" + str(i) + "@cc@" + str(j) + "@cc@" + str(k) + "@cc@" + str(l) + "@cc@" + str(m) + "@cc@" + str(n) + "@cc@" + str(deal_null(r))))

    #match_singer = raw_data.filter(lambda x: x[7] == "singer").map(lambda (a, b, c, d, e, f, g, h, i, j, k, l, m): (e, (a, f, g, h, i, j, k, l, m))) \
        #.leftOuterJoin(singer_name).map(lambda (e, ((a, f, g, h, i, j, k, l, m), n)): (str(e) + "@cc@" + str(j), (a, e, f, g, h, i, j, k, l, m, n))).leftOuterJoin(song_click1) \
        #.map(lambda (y, ((a, e, f, g, h, i, j, k, l, m, n), r)): (str(a) + "@cc@" + str(e) + "@cc@" + str(f) + "@cc@" + str(g) + "@cc@" + str(h) + "@cc@" + str(i) + "@cc@" + str(j) + "@cc@" + str(k) + "@cc@" + str(l) + "@cc@" + str(m) + "@cc@" + str(n) + "@cc@"  + str(deal_null(r))))

    #match_playlist = raw_data.filter(lambda x: x[7] == "album").map(lambda (a, b, c, d, e, f, g, h, i, j, k, l, m): (d, (a, f, g, h, i, j, k, l, m))) \
        #.leftOuterJoin(song_name).map(lambda (d, ((a, f, g, h, i, j, k, l, m), n)): (str(d) + "@cc@" + str(j), (a, d, f, g, h, i, j, k, l, m, n))).leftOuterJoin(song_click1) \
        #.map(lambda (y, ((a, d, f, g, h, i, j, k, l, m, n), r)): (str(a) + "@cc@" + str(d) + "@cc@" + str(f) + "@cc@" + str(g) + "@cc@" + str(h) + "@cc@" + str(i) + "@cc@" + str(j) + "@cc@" + str(k) + "@cc@" + str(l) + "@cc@" + str(m) + "@cc@" + str(n)  + "@cc@" + str(deal_null(r))))

    song_path = 'cosn://starmaker-research/yarou.xu/solo_data_rdd/train/song/%s'
    #singer_path = 'cosn://starmaker-research/yarou.xu/solo_data_rdd/train/singer/%s'
    #album_path = 'cosn://starmaker-research/yarou.xu/solo_data_rdd/train/album/%s'
    #playlist_path = 'cosn://starmaker-research/yarou.xu/solo_data_rdd/train/playlist/%s'
    match_song.repartition(1).saveAsTextFile(song_path % date)
    #match_singer.repartition(1).saveAsTextFile(singer_path % date)
    #match_album.repartition(1).saveAsTextFile(album_path % date)
    #match_playlist.repartition(1).saveAsTextFile(playlist_path % date)

    spark.stop()




    #spark-submit --queue B get_feature.py 20180613 a

