from collections import Counter


class BinaryPattern:

    uniform_patterns = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96,
                        112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
                        224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]

    # initialized with 3*3 matrix
    def __init__(self, input):
        # numbers which have at maximum 2 0 to one or 1 to 0 transitions in their binary representation
        self.matrix = input
        self.pattern = [self.pixel_value((0, 0)), self.pixel_value((0, 1)), self.pixel_value((0, 2)), self.pixel_value((1, 2)), self.pixel_value((2, 2)), self.pixel_value((2, 1)), self.pixel_value((2, 0)), self.pixel_value((1, 0))]

    def pixel_value(self, pixel):
        # check whether the pixel is higher than the centered pixel
        if self.matrix[pixel] > self.matrix[1, 1]:
            return 1
        else:
            return 0

    def is_uniform(self):
        # is this pattern a uniform one?
        return int(self) in self.uniform_patterns

    def __int__(self):
        # convert binary to decimal
        n = 0
        exponent = 1
        for i in self.pattern:
            n += i * exponent
            exponent *= 2
        return n


class BinaryPatternAnalysis:

    # TODO: make resistant to blocksizes which do not fit the image exactly
    def __init__(self, input, blocksize):
        self.matrix = input
        blockslength = (self.matrix.shape[0] // blocksize[0]) * (self.matrix.shape[1] // blocksize[1])
        self.blocks = [[] for b in range(blockslength)]

        # -2 to set the top left corner at maximum 3 pixels away from the edge
        for pos_y in range(self.matrix.shape[1] - 2):
            for pos_x in range(self.matrix.shape[0] - 2):
                # init pattern with 3*3 matrix with left top corner at current position
                pattern = BinaryPattern(self.matrix[pos_x:pos_x+3, pos_y:pos_y+3])
                # some cool math to get the index in 1 dim blocks list by the 2 dim position
                block = (pos_x // blocksize[0]) + (self.matrix.shape[0] // blocksize[0]) * (pos_y // blocksize[1])
                self.blocks[block].append(pattern)

    def get_histogram(self):
        histogram_vector = []
        for block in self.blocks:
            cnt = Counter({key: 0 for key in BinaryPattern.uniform_patterns})
            # the counting key for non uniform patterns:
            cnt[256] = 0

            for pattern in block:
                if pattern.is_uniform():
                    cnt[int(pattern)] += 1
                else:
                    # count patterns which are not uniform as the 256th pattern (which does not exist obviously)
                    cnt[256] += 1
            for occurrence in sorted(cnt):
                histogram_vector.append(cnt[occurrence])
        return histogram_vector
