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

if __name__ == "__main__":
    spark, sc = init_spark('get_comment_dif_group')
    day = str(sys.argv[1])

    SQL_get_user = '''
    SELECT distinct user_id, country_code
    FROM starmaker_emr.user_last_active
    WHERE country_code = "IN" AND dt = %s 
    ''' % day

    SQL_get_comment = '''
    SELECT user_id, recording_id, comment, id
    FROM starmaker_emr.comment
    WHERE parent_id = 0 AND dt = %s
    ''' % day

    SQL_get_comment_parent = '''
        SELECT distinct parent_id, dt
        FROM starmaker_emr.comment
        WHERE parent_id != 0 AND dt = %s
        ''' % day

    SQL_get_recording1 = '''
    SELECT id, user_id
    FROM starmaker_emr.recording
    WHERE dt = %s AND media_type = 1
    ''' % day
    SQL_get_recording2 = '''
        SELECT id, user_id
        FROM starmaker_emr.recording
        WHERE dt = %s AND media_type = 2
        ''' % day
    SQL_get_recording5 = '''
        SELECT id, user_id
        FROM starmaker_emr.recording
        WHERE dt = %s AND media_type = 5
        ''' % day
    SQL_get_recording6 = '''
        SELECT id, user_id
        FROM starmaker_emr.recording
        WHERE dt = %s AND media_type = 6
        ''' % day
    SQL_get_recording7 = '''
            SELECT id, user_id
            FROM starmaker_emr.recording
            WHERE dt = %s AND media_type = 7
            ''' % day
    SQL_get_recording8 = '''
            SELECT id, user_id
            FROM starmaker_emr.recording
            WHERE dt = %s AND media_type = 8
            ''' % day
    user_IN = spark.sql(SQL_get_user).rdd.filter(lambda x : x[0] >> 48 != 5).map(lambda x: (x[0],x[1]))
    comment_all = spark.sql(SQL_get_comment).rdd.map(lambda (a,b,c,d): (a,(b, c, d)))
    parent_id = spark.sql(SQL_get_comment_parent).rdd.map(lambda x: (x[0], x[1]))
    comment_india = comment_all.join(user_IN).map(lambda (a,((b,c,d),e)): (d,(a,b,c))).join(parent_id).map(lambda (a,((b,c,d),e)): (c,(b,d)))

    # 独唱-音频
    recording1 = spark.sql(SQL_get_recording1).rdd.map(lambda x: (x[0], x[1]))
    comment1 = comment_india.join(recording1).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)
    # 独唱-视频
    recording2 = spark.sql(SQL_get_recording2).rdd.map(lambda x: (x[0], x[1]))
    comment2 = comment_india.join(recording2).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)
    # 发起的音频合唱
    recording5 = spark.sql(SQL_get_recording5).rdd.map(lambda x: (x[0], x[1]))
    comment5 = comment_india.join(recording5).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)
    # 发起的视频合唱
    recording6 = spark.sql(SQL_get_recording6).rdd.map(lambda x: (x[0], x[1]))
    comment6 = comment_india.join(recording6).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)
    # 参加的音频合唱
    recording7 = spark.sql(SQL_get_recording7).rdd.map(lambda x: (x[0], x[1]))
    comment7 = comment_india.join(recording7).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)
    # 参加的视频合唱
    recording8 = spark.sql(SQL_get_recording8).rdd.map(lambda x: (x[0], x[1]))
    comment8 = comment_india.join(recording8).filter(lambda (a, ((b, c), d)): b != d).map(lambda (a, ((b, c), d)): c)

    data_file = 'cosn://starmaker-research/yarou.xu/starmaker_data_rdd/comment/%s'
    comment1.repartition(1).saveAsTextFile(data_file % day + str('1'))
    comment2.repartition(1).saveAsTextFile(data_file % day+ str('2'))
    comment5.repartition(1).saveAsTextFile(data_file % day + str('5'))
    comment6.repartition(1).saveAsTextFile(data_file % day + str('6'))
    comment7.repartition(1).saveAsTextFile(data_file % day + str('7'))
    comment8.repartition(1).saveAsTextFile(data_file % day + str('8'))

    spark.stop()

    # spark-submit --queue B get_comment_dif_group.py 20180730