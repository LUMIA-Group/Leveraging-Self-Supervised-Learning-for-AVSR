import h5py
import numpy as np

from config import args


def main():
    np.random.seed(args["SEED"])
    h5 = h5py.File("../" + args["HDF5_FILE"], "r")
    f = h5py.File("../" + args["NOISE_FILE"], "w")
    noise = np.empty(0)
    while len(noise) < 16000 * 3600:
        print("%08d / 57,600,000" % len(noise))
        noisePart = np.zeros(16000 * 60)
        indices = np.random.randint(0, len(h5["flac"]), 20)
        for ix in indices:
            audio = np.array(h5["flac"][ix])
            audio = audio / np.max(np.abs(audio))
            pos = np.random.randint(0, abs(len(audio) - len(noisePart)) + 1)
            if len(audio) > len(noisePart):
                noisePart = noisePart + audio[pos:pos + len(noisePart)]
            else:
                noisePart = noisePart[pos:pos + len(audio)] + audio
        noise = np.concatenate([noise, noisePart], axis=0)
    noise = noise[:16000 * 3600]
    noise = noise / 20
    noise = (noise - noise.mean()) / noise.std()
    noise_dest = f.create_dataset('noise', (1,), dtype=h5py.vlen_dtype(np.dtype('float32')))
    noise_dest[0] = noise

    print("\nNoise file generated.")
    f.close()
    h5.close()


if __name__ == "__main__":
    main()
