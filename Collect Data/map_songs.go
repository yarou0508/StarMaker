package main

import (
	"flag"
	"net/http"

	"github.com/BurntSushi/toml"
	"github.com/facebookgo/grace/gracehttp"
	"github.com/gin-gonic/gin"
	"runtime"

	common_conf "common/config"
	"common/helper/db_helper"
	"common/helper/es_helper"
	"common/helper/redis_helper"
	"common/service/recommender_service"
	"common/service/search_service"
	"golib/tlog"
	"lib/conf"
	"strconv"
	"strings"
	"fmt"
	"io/ioutil"
	"os"
	"bufio"
)

var (
	config = flag.String("config", "/home/worker/go_server/config/config.toml", "config file")
)

type Config struct {
	Host   string           `toml:"server_host"`
	ESHost []string         `toml:"es_host"`
	Log    tlog.Config      `toml:"Log"`
	Redis  conf.RedisConfig `toml:"Redis"`
	Mysql  conf.MysqlConfig `toml:"Mysql"`
}

func NewConfig(configFile string) (*Config, error) {
	var c Config
	_, err := toml.DecodeFile(configFile, &c)
	return &c, err
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	flag.Parse()
	c, err := NewConfig(*config)
	if err != nil {
		panic(err)
	}
	tlog.Init(c.Log)
	err = es_helper.InitEsHelper(c.ESHost)
	if err != nil {
		tlog.Fatal(err)
		panic(err)
	}
	err = redis_helper.InitRedisHelper(&c.Redis)
	if err != nil {
		tlog.Fatal(err)
		panic(err)
	}
	err = db_helper.InitDbHelper(&c.Mysql)
	if err != nil {
		tlog.Fatal(err)
		panic(err)
	}
	err = recommender_service.InitRecommender()
	if err != nil {
		tlog.Fatal(err)
		panic(err)
	}
	fmt.Println("start")
	GetTrainData()
	println("done")
	// handler := GetHandler()
	// // support graceful restart
	// gracehttp.Serve(&http.Server{Addr: c.Host, Handler: handler})
}


func GetTrainData() {
	searcher := search_service.NewSearcher()
	filePath := "/home/worker/yarou.xu/starmaker-research/ori_data.txt"
	b, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	s := string(b)
	pageP := es_helper.PageParam{0, 100}
	outputFile, outputError := os.OpenFile("/home/worker/yarou.xu/train_data.txt", os.O_WRONLY|os.O_CREATE, 0666)
	if outputError != nil {
		fmt.Printf("An error occurred with file opening or creation\n")
		return
	}
	defer outputFile.Close()
	outputWriter := bufio.NewWriter(outputFile)
	count0 := 0
	count1 := 0
	count := 0
	count2 := 0
	for _, lineStr := range strings.Split(s, "\n") {
		lineStr = strings.TrimSpace(lineStr)
		listStr := strings.Split(lineStr, "@+@")
		if len(listStr) == 11 {
			user_id := listStr[0]
			song_id := listStr[1]
			album_id := listStr[2]
			playlist_id := listStr[3]
			singer_id := listStr[4]
			action := listStr[5]
			search_keyword := listStr[6]
			search_type := listStr[7]
			timestamp := listStr[8]
			dt := listStr[9]
			order := listStr[10]
			if search_type == "songs" {
				search_type = "song"
			} else if search_type == "albums" {
				search_type = "album"
			} else if search_type == "singers" {
				search_type = "singer"
			}
			result := searcher.Search(search_keyword, search_type, pageP)
			lens := 0
			if action == "click" {
				lens = len(result)
			} else if action == "show" {
				if len(result) > 10 {
					lens = 10
				} else {
					lens = len(result)
				}
			}
			count2 += lens
			if search_type == "song" {
				id := song_id
					for i := 0; i < lens; i++ {
						count += 1
						ri := strconv.FormatInt(result[i], 10)
						if i == 0 {
							if ri == singer_id || ri == album_id {
								break
							}
						} else {
							song_id = ri
							if ri != id {
								data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "0" + "@cc@" + order + "\n"
								outputWriter.WriteString(data)
								outputWriter.Flush()
								count0 += 1
								// fmt.Println(data)
							} else if ri == id {
								data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "1" + "@cc@" + order + "\n"
								outputWriter.WriteString(data)
								outputWriter.Flush()
								count1 += 1
								if i > 10 {
									break
								}
								// fmt.Println(data)
							}
						}
					}
			} else if search_type == "album" {
				id := album_id
				for i := 0; i < lens; i++ {
					count += 1
					ri := strconv.FormatInt(result[i], 10)
					album_id = ri
					if ri != id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "0" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count0 += 1
						// fmt.Println(data)
					} else if ri == id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "1" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count1 += 1
						// fmt.Println(data)
						break
					}
				}
			} else if search_type == "playlist" {
				id := playlist_id
				for i := 0; i < lens; i++ {
					count += 1
					ri := strconv.FormatInt(result[i], 10)
					playlist_id = ri
					if ri != id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "0" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count0 += 1
						// fmt.Println(data)
					} else if ri == id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "1" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count1 += 1
						// fmt.Println(data)
						break
					}
				}
			} else if search_type == "singer" {
				id := singer_id
				for i := 0; i < lens; i++ {
					count += 1
					ri := strconv.FormatInt(result[i], 10)
					singer_id = ri
					if ri != id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "0" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count0 += 1
						// fmt.Println(data)
					} else if ri == id {
						data := user_id + "@cc@" + song_id + "@cc@" + album_id + "@cc@" + playlist_id + "@cc@" + singer_id + "@cc@" + action + "@cc@" + search_keyword + "@cc@" + search_type + "@cc@" + timestamp + "@cc@" + dt + "@cc@" + strconv.Itoa(i) + "@cc@" + "1" + "@cc@" + order + "\n"
						outputWriter.WriteString(data)
						outputWriter.Flush()
						count1 += 1
						// fmt.Println(data)
						break
					}
				}
			}
		}
	}
	fmt.Println(count0)
	fmt.Println(count1)
	fmt.Println(count)
	fmt.Println(count2)
}