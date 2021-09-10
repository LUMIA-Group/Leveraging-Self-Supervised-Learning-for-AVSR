import h5py
import numpy as np

from config import args


def main():
    np.random.seed(args["SEED"])
    h5 = h5py.File("../" + args["HDF5_FILE"], "r")
    f = h5py.File("../" + args["HUMAN_NOISE_FILE"], "w")
    noise = np.empty(0)
    while len(noise) < 16000 * 3600:
        print("%08d / 57,600,000" % len(noise))
        indices = np.random.randint(0, len(h5["flac"]))
        audio = np.array(h5["flac"][indices])
        while len(audio) < 16000:
            indices = np.random.randint(0, len(h5["flac"]))
            audio = np.array(h5["flac"][indices])

        pos = np.random.randint(0, len(audio) - 16000 + 1)
        noise = np.concatenate([noise, audio[pos:pos + 16000]], axis=0)

    noise = noise[:16000 * 3600]
    noise = (noise - noise.mean()) / noise.std()
    noise_dest = f.create_dataset('noise', (1,), dtype=h5py.vlen_dtype(np.dtype('float32')))
    noise_dest[0] = noise

    print("\nNoise file generated.")
    f.close()
    h5.close()


if __name__ == "__main__":
    main()
