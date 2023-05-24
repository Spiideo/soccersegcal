from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image, ImageReadMode

from torchvision.transforms.functional import resize, hflip
import torch
import json

from sncalib.soccerpitch import SoccerPitch
from vi3o.image import imviewsc, imview, imread, imscale
from vi3o import viewsc, view, flipp
import numpy as np
import cv2
from shapely.geometry import LineString, Point
from shapely.ops import split
from sncalib.detect_extremities import join_points


class Line:
    def __init__(self, pkts, image_shape):
        self.image_shape = image_shape
        self.missing = False
        pkts = [np.array((p['x'] * image_shape[1], p['y'] * image_shape[0])) for p in pkts]
        pkts = np.array(join_points(pkts, float('Inf'))[0])
        self.original = LineString(pkts)
        p1 = self.connect_to_image_border(pkts[0], self.direction(pkts))
        p2 = self.connect_to_image_border(pkts[-1], self.direction(pkts[::-1]))
        self.line_string = LineString(np.vstack([[p1], pkts, [p2]]))
        self.padding = [LineString([p1, pkts[0]]), LineString([pkts[-1], p2])]

    def direction(self, pkts):
        i = 1
        while i < len(pkts) - 1 and np.linalg.norm(pkts[0] - pkts[i]) < 10:
            i += 1
        return pkts[0] - pkts[i]

    def direction_at(self, pkts, pos):
        if np.linalg.norm(pkts[-1] - pos) < np.linalg.norm(pkts[0] - pos):
            pkts = pkts[::-1]
        return self.direction(pkts)

    def connect(self, other):
        if other.missing:
            return
        if self.line_string.intersects(other.line_string):
            self_parts = split(self.line_string, other.line_string)
            other_parts = split(other.line_string, self.line_string)
            pos = self.line_string.intersection(other.line_string)
            x, y = np.array(pos.coords)[0]
            if not (0 <= x < self.image_shape[1] and 0 <= y < self.image_shape[1]):
                return

            alternatives = []
            for p1 in self_parts.geoms:
                for p2 in other_parts.geoms:
                    # v1 = self.direction_at(np.array(p1.coords), pos)
                    # v2 = self.direction_at(np.array(p2.coords), pos)
                    #     if np.arctan2(np.cross(v1, v2), np.dot(v1, v2)) < 0:  # Right turn
                    alternatives.append((p1, p2))
            best = max(alternatives, key=lambda a: min(a[0].hausdorff_distance(pad) for pad in self.padding) + min(a[1].hausdorff_distance(pad) for pad in other.padding))
            self.line_string, other.line_string = best

    def connect_to_image_border(self, p, v):
        if p is None:
            return 0, None
        v = v / np.linalg.norm(v)
        tt  = [-p[0] / v[0], -p[1] / v[1], (self.image_shape[1] - 1 - p[0]) / v[0], (self.image_shape[0] - 1 - p[1]) / v[1]]
        tt = [t for t in tt if t >= 0 and np.isfinite(t)]
        if len(tt) == 0:
            return p
        t = min(tt)
        return p + t * v


class MisingLine(Line):
    def __init__(self, image_shape) -> None:
        self.image_shape = image_shape
        self.pkts = [None, None]
        self.v1 = self.v2 = None
        self.missing = True

    def connect(self, other):
        pass

    def match_direction(self, other):
        pass

class SoccerNetFieldSegmentationDataset(Dataset):
    def __init__(self, datasetpath="data/SoccerNet/calibration-2023", split="valid", width=640, height=None, skip_bad=False):
        if height is None:
            height = (width * 9) // 16
        self.root = Path(datasetpath)
        self.images = list((self.root / split).glob('*.jpg'))
        self.images.sort()
        self.shape = (height,  width)
        self.bad = {
            "valid": {11, 32, 60, 64, 71, 82, 85, 108, 129, 151, 176, 200, 280, 284, 295, 310, 315, 367, 374, 381, 407, 411, 420, 458, 459, 465, 521, 525, 527, 532, 536, 538, 544, 553, 572, 579, 592, 595, 614, 617, 638, 651, 710, 756, 780, 786, 790, 799, 801, 809, 813, 847, 873, 876, 883, 884, 914, 920, 927, 962, 965, 993, 1002, 1012, 1015, 1026, 1040, 1046, 1066, 1067, 1078, 1080, 1088, 1096, 1104, 1119, 1139, 1141, 1143, 1148, 1152, 1225, 1244, 1249, 1258, 1264, 1265, 1268, 1284, 1285, 1286, 1300, 1301, 1316, 1318, 1325, 1338, 1341, 1357, 1379, 1394, 1424, 1439, 1444, 1457, 1458, 1467, 1468, 1477, 1493, 1498, 1509, 1545, 1577, 1600, 1619, 1649, 1650, 1737, 1761, 1763, 1768, 1789, 1814, 1816, 1841, 1865, 1870, 1905, 1912, 1920, 1922, 1931, 1945, 1956, 2026, 2056, 2057, 2074, 2096, 2097, 2111, 2112, 2121, 2138, 2143, 2147, 2153, 2155, 2186, 2187, 2194, 2199, 2202, 2232, 2244, 2262, 2282, 2297, 2306, 2317, 2327, 2334, 2358, 2363, 2364, 2375, 2393, 2394, 2397, 2504, 2518, 2521, 2525, 2541, 2554, 2557, 2558, 2562, 2564, 2573, 2580, 2588, 2604, 2613, 2618, 2650, 2671, 2756, 2768, 2800, 2829, 2830, 2832, 2835, 2838, 2871, 2886, 2928, 2941, 2995, 3007, 3015, 3026, 3046, 3049, 3071, 3091, 3106, 3124, 3131, 3132, 3133, 3134, 3149, 3161, 3162, 3179, 3182, 3207},
            "test": {23, 33, 75, 92, 100, 112, 119, 129, 141, 181, 188, 258, 294, 304, 314, 321, 342, 343, 482, 487, 496, 501, 508, 518, 545, 582, 586, 589, 591, 609, 615, 646, 649, 697, 713, 747, 748, 756, 787, 829, 834, 839, 850, 863, 869, 872, 941, 984, 1007, 1020, 1025, 1028, 1030, 1039, 1042, 1071, 1086, 1098, 1115, 1118, 1121, 1125, 1131, 1136, 1158, 1169, 1191, 1223, 1224, 1227, 1245, 1262, 1268, 1269, 1279, 1281, 1307, 1311, 1323, 1326, 1328, 1341, 1357, 1394, 1413, 1414, 1500, 1546, 1584, 1616, 1631, 1640, 1644, 1663, 1698, 1700, 1702, 1707, 1759, 1763, 1788, 1795, 1817, 1886, 1889, 1907, 1930, 1934, 1970, 2051, 2058, 2062, 2070, 2072, 2130, 2173, 2181, 2185, 2188, 2208, 2216, 2268, 2276, 2280, 2286, 2294, 2305, 2307, 2317, 2330, 2346, 2389, 2439, 2453, 2462, 2493, 2517, 2527, 2564, 2570, 2592, 2608, 2621, 2627, 2631, 2637, 2647, 2708, 2713, 2732, 2786, 2850, 2862, 2888, 2899, 2902, 2920, 2925, 2962, 2963, 2972, 2998, 3014, 3029, 3033, 3039, 3040, 3046, 3053, 3056, 3057, 3060, 3064, 3077, 3082, 3086, 3089, 3104, 3122, 3137},
            "train": {81, 87, 89, 109, 128, 153, 154, 186, 195, 198, 227, 274, 280, 284, 288, 296, 300, 336, 339, 340, 350, 351, 372, 395, 400, 402, 409, 419, 426, 431, 432, 433, 435, 451, 544, 609, 644, 658, 711, 723, 724, 794, 840, 877, 901, 903, 972, 983, 1004, 1036, 1081, 1087, 1089, 1110, 1257, 1339, 1342, 1370, 1498, 1516, 1518, 1522, 1532, 1563, 1600, 1643, 1645, 1655, 1666, 1678, 1706, 1712, 1739, 1753, 1844, 1919, 1931, 1937, 1961, 1968, 2029, 2037, 2041, 2052, 2056, 2085, 2115, 2125, 2168, 2236, 2244, 2260, 2261, 2280, 2287, 2307, 2328, 2346, 2367, 2369, 2373, 2380, 2410, 2426, 2443, 2465, 2466, 2511, 2543, 2591, 2629, 2631, 2651, 2656, 2661, 2674, 2692, 2694, 2698, 2707, 2719, 2730, 2740, 2744, 2747, 2751, 2753, 2766, 2776, 2780, 2812, 2821, 2860, 2890, 2930, 2969, 2971, 2973, 2981, 2997, 3019, 3052, 3082, 3168, 3186, 3218, 3247, 3264, 3278, 3366, 3385, 3392, 3434, 3451, 3460, 3486, 3535, 3539, 3556, 3564, 3660, 3662, 3670, 3696, 3743, 3798, 3825, 3841, 3871, 3873, 3894, 3930, 3947, 3948, 3968, 3987, 4005, 4017, 4039, 4053, 4061, 4063, 4102, 4113, 4123, 4144, 4147, 4157, 4178, 4191, 4204, 4216, 4254, 4264, 4273, 4294, 4316, 4337, 4355, 4364, 4386, 4396, 4435, 4485, 4493, 4575, 4599, 4605, 4628, 4672, 4688, 4744, 4758, 4778, 4783, 4784, 4786, 4826, 4861, 4894, 4896, 4902, 4909, 4918, 4923, 4928, 4929, 4932, 4934, 4958, 4959, 4960, 4976, 4977, 4984, 5018, 5035, 5045, 5048, 5053, 5066, 5067, 5070, 5073, 5087, 5089, 5142, 5161, 5169, 5172, 5190, 5194, 5195, 5240, 5241, 5268, 5281, 5293, 5303, 5319, 5360, 5365, 5394, 5395, 5415, 5416, 5446, 5481, 5484, 5528, 5575, 5581, 5626, 5670, 5691, 5721, 5732, 5746, 5762, 5780, 5786, 5796, 5806, 5814, 5927, 5946, 6135, 6150, 6156, 6157, 6164, 6168, 6178, 6203, 6246, 6247, 6250, 6278, 6279, 6315, 6316, 6321, 6323, 6338, 6371, 6388, 6393, 6396, 6404, 6425, 6459, 6462, 6486, 6508, 6518, 6522, 6523, 6553, 6565, 6600, 6614, 6617, 6665, 6674, 6712, 6727, 6756, 6765, 6775, 6781, 6787, 6812, 6855, 6892, 6936, 6940, 6943, 6944, 6955, 6959, 6969, 6981, 6998, 7008, 7025, 7041, 7049, 7061, 7064, 7065, 7092, 7121, 7128, 7146, 7155, 7161, 7164, 7218, 7224, 7241, 7251, 7276, 7290, 7293, 7294, 7299, 7310, 7315, 7334, 7338, 7343, 7344, 7432, 7441, 7442, 7472, 7475, 7529, 7538, 7548, 7554, 7590, 7592, 7623, 7624, 7632, 7644, 7692, 7695, 7702, 7710, 7712, 7719, 7745, 7757, 7781, 7787, 7832, 7845, 7857, 7866, 7867, 7869, 7885, 7896, 7962, 7991, 7996, 8012, 8052, 8085, 8115, 8133, 8137, 8142, 8144, 8158, 8164, 8190, 8214, 8215, 8222, 8250, 8262, 8280, 8312, 8329, 8336, 8338, 8350, 8351, 8378, 8445, 8450, 8455, 8457, 8461, 8474, 8476, 8486, 8499, 8506, 8507, 8516, 8525, 8534, 8556, 8565, 8568, 8571, 8584, 8590, 8595, 8597, 8598, 8638, 8675, 8688, 8700, 8705, 8713, 8732, 8733, 8859, 8881, 8903, 8922, 8924, 8929, 8930, 8952, 8991, 9033, 9040, 9078, 9122, 9158, 9159, 9176, 9198, 9219, 9256, 9261, 9275, 9287, 9371, 9396, 9397, 9423, 9451, 9452, 9455, 9462, 9469, 9482, 9527, 9533, 9538, 9568, 9635, 9640, 9654, 9664, 9667, 9689, 9694, 9718, 9728, 9740, 9750, 9753, 9768, 9771, 9824, 9869, 9876, 9888, 9892, 9894, 9896, 9922, 9923, 9993, 10039, 10067, 10076, 10103, 10104, 10113, 10129, 10157, 10159, 10183, 10193, 10204, 10233, 10237, 10273, 10277, 10282, 10294, 10309, 10311, 10319, 10321, 10336, 10346, 10358, 10363, 10395, 10419, 10455, 10468, 10469, 10471, 10480, 10523, 10534, 10557, 10562, 10593, 10626, 10627, 10686, 10705, 10712, 10724, 10745, 10797, 10799, 10804, 10820, 10821, 10822, 10827, 10860, 10865, 10873, 10874, 10875, 10882, 10916, 10951, 10964, 10973, 10991, 11039, 11054, 11076, 11086, 11098, 11113, 11126, 11137, 11172, 11181, 11219, 11221, 11223, 11237, 11243, 11246, 11247, 11253, 11262, 11304, 11327, 11345, 11346, 11348, 11356, 11366, 11397, 11400, 11411, 11415, 11478, 11513, 11543, 11561, 11569, 11583, 11590, 11597, 11614, 11621, 11640, 11650, 11666, 11677, 11680, 11695, 11700, 11703, 11716, 11731, 11755, 11758, 11764, 11809, 11816, 11817, 11861, 11862, 11878, 11887, 11918, 11946, 11951, 11959, 11975, 11980, 11993, 12008, 12011, 12016, 12044, 12063, 12077, 12093, 12095, 12108, 12113, 12131, 12173, 12177, 12178, 12183, 12184, 12205, 12239, 12248, 12277, 12284, 12285, 12286, 12300, 12318, 12327, 12329, 12333, 12336, 12354, 12357, 12377, 12379, 12380, 12394, 12395, 12399, 12400, 12403, 12406, 12419, 12425, 12431, 12440, 12444, 12482, 12497, 12503, 12507, 12523, 12529, 12540, 12554, 12581, 12611, 12648, 12694, 12727, 12743, 12838, 12850, 12856, 12858, 12870, 12875, 12889, 12890, 12927, 12937, 12949, 12985, 13035, 13050, 13089, 13111, 13128, 13146, 13153, 13168, 13213, 13391, 13398, 13400, 13404, 13408, 13414, 13434, 13492, 13503, 13508, 13541, 13542, 13577, 13578, 13584, 13588, 13649, 13651, 13752, 13756, 13757, 13767, 13771, 13792, 13818, 13847, 13853, 13858, 13879, 13883, 13889, 13896, 13904, 13919, 13928, 13960, 13962, 13973, 13981, 14010, 14021, 14024, 14029, 14076, 14107, 14142, 14182, 14188, 14189, 14201, 14208, 14217, 14229, 14306, 14313, 14317, 14335, 14354, 14357, 14378, 14381, 14414, 14486, 14494, 14509, 14511, 14528, 14529, 14536, 14538, 14550, 14554, 14563, 14575, 14580, 14583, 14601, 14616, 14647, 14657, 14659, 14704, 14756, 14779, 14783, 14812, 14824, 14832, 14842, 14852, 14863, 14869, 14881, 14894, 14901, 14911, 14981, 14997, 15008, 15009, 15023, 15060, 15067, 15070, 15073, 15087, 15090, 15098, 15100, 15107, 15110, 15111, 15117, 15130, 15141, 15154, 15162, 15167, 15170, 15175, 15210, 15212, 15229, 15233, 15241, 15248, 15261, 15262, 15270, 15297, 15315, 15318, 15324, 15339, 15342, 15364, 15368, 15377, 15383, 15392, 15400, 15402, 15403, 15427, 15437, 15471, 15472, 15496, 15498, 15516, 15517, 15521, 15529, 15542, 15545, 15550, 15560, 15564, 15571, 15572, 15578, 15630, 15635, 15660, 15693, 15709, 15718, 15740, 15743, 15755, 15812, 15822, 15831, 15853, 15858, 15860, 15869, 15877, 15884, 15890, 15893, 15895, 15899, 15904, 15919, 15921, 15938, 15970, 15973, 15986, 15998, 15999, 16040, 16070, 16072, 16076, 16091, 16098, 16115, 16128, 16169, 16172, 16173, 16185, 16189, 16191, 16193, 16197, 16203, 16211, 16232, 16233, 16247, 16250, 16255, 16262, 16267, 16305, 16309, 16315, 16320, 16327, 16342, 16356, 16362, 16373, 16380, 16392},
            "challenge": set(),
        }[split]
        self.indexes = list(range(len(self.images)))
        if skip_bad:
            self.indexes = [i for i in self.indexes if i not in self.bad]
            self.images = [img for i, img in enumerate(self.images) if i not in self.bad]
        self.class_names = ['FullField', 'CircleCentral', 'BigRect', 'CircleSide', 'SmallRect', 'Goal']
        self.split = split

    def __len__(self):
        return len(self.images)

    def lines(self, index):
        fn = self.images[index]
        file = fn.parent / fn.name.replace('.jpg', '.json')
        if not file.exists():
            return None
        with file.open() as fp:
            lines = json.load(fp)
        return {n: l for n, l in lines.items() if len(l) > 1}

    def baseline_camera(self, index):
        fn = self.root / 'good_baseline_cameras' / self.split / self.images[index].name.replace('.jpg', '.json')
        if fn.exists():
            with fn.open() as fd:
                return json.load(fd)
        return None

    def show_lines(self, index, pasue=True):
        img = imread(self.images[index])
        img = imscale(img, (self.shape[1], self.shape[0]))
        for n, points in self.lines(index).items():
            print(n)
            pkts = [np.array((p['x'] * img.shape[1], p['y'] * img.shape[0])) for p in points]
            pkts = np.array(join_points(pkts, float('Inf'))[0])
            if 'Goal left' in n or ('Goal' not in n and 'left' in n):
                c1, c2 = 255, 0
            elif 'Goal right' in n or ('Goal' not in n and 'right' in n):
                c1, c2 = 0, 255
            else:
                c1, c2 = 128, 128
            if 'Side' in n:
                c3 = 128
            else:
                c3 = 0
            cv2.polylines(img, [np.int32(pkts)], False, (c1,c2,c3), 3)
        view(img, pause=pasue)

    def __getitem__(self, index):
        fn = self.images[index]
        img = read_image(str(fn), ImageReadMode.RGB)
        img = resize(img, self.shape)
        lines = self.lines(index)

        dt = torch.get_default_dtype()
        if lines is None:
            return dict(
                image=img.to(dt).div(255),
                name=fn.name,
            )


        segments = np.zeros((6,) + self.shape, np.uint8)
        for name, area in SoccerPitch.field_areas.items():
            all_lines = [Line(lines[n], self.shape).original.coords
                         for n in area['contains'] if n in lines]
            if len(all_lines) == 0:
                continue
            center_x, center_y = map(int, np.round(np.vstack(all_lines).mean(0)))
            center_x = min(max(center_x, 0), self.shape[1] - 1)
            center_y = min(max(center_y, 0), self.shape[0] - 1)

            rim = []
            for n in area['border']:
                if n in lines:
                    rim.append(Line(lines[n], self.shape))
                else:
                    rim.append(MisingLine(self.shape))

            # for i in range(len(rim)):
            #     rim[i-1].connect(rim[i])

            segs = np.zeros(self.shape, np.uint8)
            if SEG_DEBUG:
                print(lines.keys())
                segs[:] = img[0].numpy() // 2
            plot_lines = [np.int32(l.line_string.coords) for l in rim if not l.missing]
            cv2.polylines(segs, plot_lines, False, 255, 1)

            if segs[max(center_y-1, 0):min(center_y+2, self.shape[0]), max(center_x-1, 0):min(center_x+2, self.shape[1])].max() == 255:  # If the center is on the line try moving it towards the long border not contained
                for n in area['border']:
                    if n not in area['contains'] and n in lines:
                        l = Line(lines[n], self.shape).line_string
                        p = np.array(l.interpolate(l.project(Point([center_x, center_y]))).coords)[0]
                        p -= (center_x, center_y)
                        l = np.linalg.norm(p)
                        if l > 0:
                            p /= l
                            center_x, center_y = map(int, (center_x, center_y) + 2 * p)
                            center_x = min(max(center_x, 0), self.shape[1] - 1)
                            center_y = min(max(center_y, 0), self.shape[0] - 1)

            if SEG_DEBUG:
                cv2.circle(segs, (center_x, center_y), 3, 255, -1)
                pad_lines = [np.int32(pad.coords) for l in rim if not l.missing for pad in l.padding]
                cv2.polylines(segs, pad_lines, False, 128, 1)
                imviewsc(segs)
                segs = np.zeros(self.shape, np.uint8)
                cv2.polylines(segs, plot_lines, False, 255, 1)
            cv2.floodFill(segs, None, (center_x, center_y), 255)
            cv2.polylines(segs, plot_lines, False, 0, 1)
            # segs = cv2.dilate(segs, np.ones((3,3), np.uint8))
            if SEG_DEBUG:
                imviewsc(segs)
            segments[area['index']][segs>0] = segs[segs>0]

            # from vi3o import view
            # view(segments)
            # view(img.numpy()[0])

        # Connectc regions sharing borders - mostly estetic - introduces unwanted overlaps
        # segments[5] = cv2.dilate(segments[5], np.ones((3,3), np.uint8))
        # segments[3] = cv2.dilate(segments[3], np.ones((3,3), np.uint8))

        return dict(
            image=img.to(dt).div(255),
            segments=torch.tensor(segments).to(dt).div(255),
            name=fn.name,
        )

    def find_closest_image_border(self, pkt):
        bottom = self.shape[0] - 1
        right = self.shape[1] - 1
        dists = np.abs([pkt[0], pkt[1], right - pkt[0], bottom - pkt[1]])
        positions = [(0, pkt[1]), (pkt[0], 0), (right, pkt[1]), (pkt[0], bottom)]
        self.corners = [(0, bottom), (0, 0), (right, 0), (bottom, right)]
        i = np.argmin(dists)
        return positions[i]

class HFlipDataset(Dataset):
    def __init__(self, parent) -> None:
        self.parent = parent
        self.class_names = parent.class_names

    def __len__(self):
        return 2 * len(self.parent)

    def __getitem__(self, index):
        item = self.parent[index // 2]
        if index & 1:
            item['segments'] = hflip(item['segments'])
            item['image'] = hflip(item['image'])
        return item

def pview(img, pause=True):
    img = img.detach().cpu().numpy()
    if len(img.shape) == 3:
        img = img.transpose(1, 2, 0)
    viewsc(img, pause=pause)

SEG_DEBUG = False

if __name__ == '__main__':
    # data = HFlipDataset(SoccerNetFieldSegmentationDataset(split='valid'))
    data = SoccerNetFieldSegmentationDataset(split='valid')

    flipp()
    for i in range(10, len(data)):
        print(i)
        data.show_lines(i)
        entry = data[i]
        pview(entry['image'])
        segs = entry['segments']
        pview(segs[:3]/2 + segs[3:]/2)
        flipp(pause=True)
