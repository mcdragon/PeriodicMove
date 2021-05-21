import pickle
from sklearn.cluster import KMeans
from pathlib import Path
import math
import matplotlib.pyplot as plt


class Region:
    def __init__(self, lon_range, lat_range, delta=0.001):
        self.lat_min = lat_range[0]
        self.lat_max = lat_range[1]
        self.lon_min = lon_range[0]
        self.lon_max = lon_range[1]
        self.delta = delta
        # init grids
        self.x = int((self.lon_max-self.lon_min)/self.delta)+1
        self.y = int((self.lat_max-self.lat_min)/self.delta)+1

    def lonlat_to_id(self, lon, lat):
        lon_index = int((lon - self.lon_min)/self.delta)
        lat_index = int((lat - self.lat_min)/self.delta)
        return lat_index*self.x+lon_index

    def id_to_xy(self, gid):
        lat_index = gid // self.x
        lon_index = gid % self.x
        return lon_index, lat_index

    def id_to_lonlat(self, gid):
        lat_index = gid // self.x
        lon_index = gid % self.x
        delta_lon = lon_index * self.delta + self.delta/2
        delta_lat = lat_index * self.delta + self.delta/2
        return self.lon_min+delta_lon, self.lat_min+delta_lat

    def point_to_id(self, point):
        point.id = self.lonlat_to_id(point.lon, point.lat)

    def dis_between_two_gid(self, gid1, gid2):
        gid1_lon, gid1_lat = self.id_to_lonlat(gid1)
        gid2_lon, gid2_lat = self.id_to_lonlat(gid2)
        return math.sqrt(math.pow((gid1_lon-gid2_lon)*100000,2) + math.pow((gid1_lat-gid2_lat)*100000,2))

    def __repr__(self):
        return "lon_num(x): {}, lat_num(y): {}".format(self.x, self.y)

beijing_lon_range = [138.4, 139.8]
beijing_lat_range = [34.8, 36.2]
delta = 0.005
region = Region(beijing_lon_range, beijing_lat_range, delta)

emb_w2i_dict_path = Path("./save_models/dataset_foursquare_hiddensize_128_nheads_4_distloss_True_dropout_0.3_alpha_0.15_lr_0.0010_nocutdistloss.emb")

embedding_matrix, w2i_dict = pickle.load(emb_w2i_dict_path.open("rb"))
i2w_dict = dict([(item[1], item[0]) for item in w2i_dict.items()])

model = KMeans(n_clusters=8, random_state=2)
clusters = model.fit_predict(embedding_matrix)

w2c = dict()
for i in i2w_dict:
    w2c[i2w_dict[i]] = clusters[i]

w2xy = dict()
for w in w2i_dict:
    try:
        w2xy[w] = region.id_to_xy(int(w))
    except:
        continue

x = []
y = []
c = []
for w in w2xy:
    x.append(w2xy[w][0])
    y.append(w2xy[w][1])
    c.append(w2c[w])
s = [2] * len(x)

plt.scatter(x, y, c=c, s=s, cmap=plt.cm.Spectral)
plt.xticks([])
plt.yticks([])
plt.savefig("figs/foursquare_cluster.pdf")
