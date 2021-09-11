args = dict()
"""
    =====Directory=====
"""
# define the "relative directory" relative to the project root dir
args["CODE_DIRECTORY"] = ''
args["DATA_DIRECTORY"] = "../LRS2/mvlrs_v1"
args["HDF5_FILE"] = "../LRS2/mvlrs_v1/LRS2.h5"
args["LRW_DATA_DIRECTORY"] = "../LRW/lipread_mp4"
args["LRW_HDF5_FILE"] = "../LRW/LRW.h5"
args["NOISE_FILE"] = "../LRS2/mvlrs_v1/Noise.h5"
args["HUMAN_NOISE_FILE"] = "../LRS2/mvlrs_v1/HumanNoise.h5"
args["SHAPE_PREDICTOR_FILE"] = "../pretrain_model/shape_predictor_68_face_landmarks.dat"
args["MOCO_FILE"] = "../pretrain_model/moco_v2_200ep_pretrain.pth"
"""
    =====Experimental Setting=====
"""
args["GPU_IDS"] = [0, 1, 2, 3]
args["GPU_ID"] = 0
args["NUM_WORKERS"] = 4
args["NUM_CPU_CORE"] = 2
args["BATCH_SIZE"] = 8
args["STEP_SIZE"] = 16384
args["INIT_LR"] = 1e-4

"""
    For LRW
"""
args["LRW_WARMUP_PERIOD"] = 80
# Used for LRW training
args["TRAIN_LRW_MODEL_FILE"] = None
# Used for LRW evaluation
args["EVAL_LRW_MODEL_FILE"] = None
"""
    For LRS2
"""
args["ALPHA"] = 0.2

args["MODAL"] = "AV"  # "AO" "VO" "AV"
args["NOISE_SNR_DB"] = 5  # noise level in dB SNR
args["NOISE_PROBABILITY"] = 0.25

args["W2V_FREEZE_EPOCH"] = 40
args["LRS2_WARMUP_PERIOD"] = 80

args["MOCO_FRONTEND_FILE"] = "../pretrain_model/moco_frontend.pt"
args["WAV2VEC_FILE"] = "../pretrain_model/wav2vec_vox_new.pt"
args["TRAIN_LRS2_MODEL_FILE"] = None
args["TRAINED_AO_FILE"] = None
args["TRAINED_VO_FILE"] = None

# Used for evaluation
args["LAMBDA"] = 0.1
args["BEAM_WIDTH"] = 5
args["TEST_WITH_NOISE"] = False
args["TEST_NOISE_SNR_DB"] = 5
args["DECODE_TYPE"] = "HYBRID"  # HYBRID ATTN TFATTN CTC
args["EVAL_LRS2_MODEL_FILE"] = None

"""
    =====Default Parameters=====
"""
# preprocessing
args["PREPROCESSING_NUM_OF_PROCESS"] = 32

# checkpoint
args["SAVE_FREQUENCY"] = 10

# data
args["MAIN_REQ_INPUT_LENGTH"] = 80  # minimum input length while training

# training
args["SEED"] = 19260817  # seed for random number generators
args["NUM_STEPS"] = 1000  # maximum number of steps to train for (early stopping is used)

# optimizer, scheduler and modality dropping
args["FINAL_LR"] = 1e-7  # final learning rate for scheduler
args["LR_SCHEDULER_WAIT"] = 40  # number of steps to wait to lower learning rate
args["LR_SCHEDULER_METRICS"] = "WER"
args["LR_SCHEDULER_FACTOR"] = 0.5  # learning rate decrease factor for scheduler
args["LR_SCHEDULER_THRESH"] = 0.001  # threshold to check plateau-ing of wer
args["MOMENTUM1"] = 0.9  # optimizer momentum 1 value
args["MOMENTUM2"] = 0.999  # optimizer momentum 2 value

# model
args["VIDEO_FEATURE_SIZE"] = 2048
args["AUDIO_FEATURE_SIZE"] = 1024
args["FRONTEND_DMODEL"] = 1024
args["FRAME_LENGTH"] = 29
args["WORD_NUM_CLASSES"] = 500
args["CHAR_NUM_CLASSES"] = 40  # number of output characters
args["PHON_NUM_CLASSES"] = 74  # number of output phonemes

# transformer architecture
args["PE_MAX_LENGTH"] = 500  # length up to which we calculate positional encodings
args["DMODEL"] = 512  # transformer input feature size
args["TX_ATTENTION_HEADS"] = 8  # number of attention heads in multihead attention layer
args["TX_NUM_LAYERS"] = 6  # number of Transformer Encoder blocks in the stack
args["TX_FEEDFORWARD_DIM"] = 2048  # hidden layer size in feedforward network of transformer
args["TX_DROPOUT"] = 0.1  # dropout probability in the transformer

# Dict
args["CHAR_TO_INDEX"] = {" ": 1, "'": 22, "1": 30, "0": 29, "3": 37, "2": 32, "5": 34, "4": 38, "7": 36, "6": 35, "9": 31, "8": 33, "A": 5, "C": 17,
                         "B": 20, "E": 2, "D": 12, "G": 16, "F": 19, "I": 6, "H": 9, "K": 24, "J": 25, "M": 18, "L": 11, "O": 4, "N": 7, "Q": 27,
                         "P": 21, "S": 8, "R": 10, "U": 13, "T": 3, "W": 15, "V": 23, "Y": 14, "X": 26, "Z": 28, "<EOS>": 39}

args["INDEX_TO_CHAR"] = {1: " ", 22: "'", 30: "1", 29: "0", 37: "3", 32: "2", 34: "5", 38: "4", 36: "7", 35: "6", 31: "9", 33: "8", 5: "A", 17: "C",
                         20: "B", 2: "E", 12: "D", 16: "G", 19: "F", 6: "I", 9: "H", 24: "K", 25: "J", 18: "M", 11: "L", 4: "O", 7: "N", 27: "Q",
                         21: "P", 8: "S", 10: "R", 13: "U", 3: "T", 15: "W", 23: "V", 14: "Y", 26: "X", 28: "Z", 39: "<EOS>"}


args["WORD_TO_INDEX"] = {'UNION': 0, 'BECOME': 1, 'COMPANIES': 2, 'NUMBERS': 3, 'CHILDREN': 4, 'TEMPERATURES': 5, 'BUILD': 6, 'ANOTHER': 7,
                         'ECONOMY': 8, 'PARTIES': 9, 'FUTURE': 10, 'SERIOUS': 11, 'SIMPLY': 12, 'FOREIGN': 13, 'SINGLE': 14, 'SEEMS': 15,
                         'INCREASE': 16, 'VICTIMS': 17, 'NEVER': 18, 'RIGHTS': 19, 'AUTHORITIES': 20, 'MONTH': 21, 'EMERGENCY': 22, 'CLOUD': 23,
                         'SOUTH': 24, 'MILLION': 25, 'SHORT': 26, 'FRANCE': 27, 'JUSTICE': 28, 'SERVICE': 29, 'AGREE': 30, 'INVOLVED': 31,
                         'COUNCIL': 32, 'RATHER': 33, 'RATES': 34, 'HAPPENING': 35, 'GUILTY': 36, 'ELECTION': 37, 'LEVEL': 38, 'REFERENDUM': 39,
                         'AFRICA': 40, 'ENOUGH': 41, 'LATER': 42, 'MEETING': 43, 'WATER': 44, 'RUSSIAN': 45, 'MARKET': 46, 'LEADER': 47, 'WANTS': 48,
                         'MIDDLE': 49, 'OFFICE': 50, 'GOING': 51, 'SCOTLAND': 52, 'MINISTER': 53, 'WARNING': 54, 'WALES': 55, 'STREET': 56,
                         'SOMETHING': 57, 'GETTING': 58, 'CHANCE': 59, 'STAND': 60, 'FOOTBALL': 61, 'SITUATION': 62, 'CANCER': 63, 'CHINA': 64,
                         'PRICES': 65, 'PUBLIC': 66, 'QUESTIONS': 67, 'SENSE': 68, 'PARENTS': 69, 'TALKING': 70, 'MEDICAL': 71, 'CRIME': 72,
                         'RETURN': 73, 'BECAUSE': 74, 'INDUSTRY': 75, 'PRIME': 76, 'PAYING': 77, 'OPERATION': 78, 'ABSOLUTELY': 79, 'FRENCH': 80,
                         'PROCESS': 81, 'TOWARDS': 82, 'ISLAMIC': 83, 'OTHERS': 84, 'LARGE': 85, 'HEAVY': 86, 'ANYTHING': 87, 'OTHER': 88,
                         'TERMS': 89, 'REMEMBER': 90, 'WEEKEND': 91, 'SCHOOLS': 92, 'DESCRIBED': 93, 'TRYING': 94, 'SYRIA': 95, 'AMERICAN': 96,
                         'GREAT': 97, 'HOSPITAL': 98, 'PERSONAL': 99, 'RECENT': 100, 'AGAINST': 101, 'AMERICA': 102, 'ORDER': 103, 'WOULD': 104,
                         'CERTAINLY': 105, 'WEEKS': 106, 'NOTHING': 107, 'CAPITAL': 108, 'LATEST': 109, 'FORCE': 110, 'LEVELS': 111, 'GROUND': 112,
                         'MEASURES': 113, 'CANNOT': 114, 'ASKING': 115, 'MANCHESTER': 116, 'LIVES': 117, 'FORWARD': 118, 'TAKING': 119, 'AGAIN': 120,
                         'EVENING': 121, 'EXAMPLE': 122, 'NIGHT': 123, 'SYRIAN': 124, 'WELCOME': 125, 'FRONT': 126, 'PATIENTS': 127, 'WATCHING': 128,
                         'BRITAIN': 129, 'CUSTOMERS': 130, 'CHANGE': 131, 'WORKING': 132, 'WHILE': 133, 'AHEAD': 134, 'NATIONAL': 135,
                         'CONFERENCE': 136, 'MAKES': 137, 'PLANS': 138, 'FACING': 139, 'CHARGE': 140, 'HOURS': 141, 'WHICH': 142, 'COUNTRY': 143,
                         'RESPONSE': 144, 'WEAPONS': 145, 'STARTED': 146, 'EVIDENCE': 147, 'DOING': 148, 'MEDIA': 149, 'TAKEN': 150, 'STORY': 151,
                         'EXACTLY': 152, 'AMOUNT': 153, 'BIGGEST': 154, 'CAMERON': 155, 'DETAILS': 156, 'HEALTH': 157, 'MASSIVE': 158,
                         'VIOLENCE': 159, 'EVERYONE': 160, 'MAKING': 161, 'LITTLE': 162, 'FAMILIES': 163, 'STILL': 164, 'MINUTES': 165, 'HEART': 166,
                         'NORTH': 167, 'EXPECT': 168, 'POLITICS': 169, 'ISSUE': 170, 'DIFFERENT': 171, 'WAITING': 172, 'COMES': 173,
                         'POLITICIANS': 174, 'CHANGES': 175, 'UNDERSTAND': 176, 'WORLD': 177, 'SECRETARY': 178, 'ANNOUNCED': 179, 'FIGHT': 180,
                         'PRESS': 181, 'BENEFIT': 182, 'FIGURES': 183, 'SUNSHINE': 184, 'SPEND': 185, 'GENERAL': 186, 'MORNING': 187, 'WRONG': 188,
                         'LONGER': 189, 'CLOSE': 190, 'AFTERNOON': 191, 'STAFF': 192, 'AFFAIRS': 193, 'INQUIRY': 194, 'DAVID': 195, 'POINT': 196,
                         'INSIDE': 197, 'SECTOR': 198, 'THIRD': 199, 'NORTHERN': 200, 'MEANS': 201, 'BUSINESSES': 202, 'CONFLICT': 203,
                         'DIFFICULT': 204, 'SCOTTISH': 205, 'THINK': 206, 'TOMORROW': 207, 'EVENTS': 208, 'MINISTERS': 209, 'COURSE': 210,
                         'BETWEEN': 211, 'ASKED': 212, 'EVERYTHING': 213, 'ARRESTED': 214, 'BUDGET': 215, 'OPPOSITION': 216, 'WEATHER': 217,
                         'PLACE': 218, 'POWER': 219, 'WORDS': 220, 'SMALL': 221, 'COULD': 222, 'DESPITE': 223, 'LABOUR': 224, 'RUNNING': 225,
                         'CONTROL': 226, 'POTENTIAL': 227, 'IRELAND': 228, 'HOUSE': 229, 'FINANCIAL': 230, 'MATTER': 231, 'THOSE': 232,
                         'COMPANY': 233, 'RUSSIA': 234, 'ENERGY': 235, 'GEORGE': 236, 'WITHOUT': 237, 'STATES': 238, 'LEAST': 239, 'SUPPORT': 240,
                         'EVERYBODY': 241, 'UNDER': 242, 'CURRENT': 243, 'PRISON': 244, 'HAVING': 245, 'DECIDED': 246, 'DEBATE': 247, 'HOMES': 248,
                         'EXPECTED': 249, 'PROBLEM': 250, 'STATEMENT': 251, 'FIRST': 252, 'BETTER': 253, 'POLICE': 254, 'FOUND': 255, 'SYSTEM': 256,
                         'AFTER': 257, 'MURDER': 258, 'BUILDING': 259, 'JUDGE': 260, 'PERHAPS': 261, 'REASON': 262, 'PROTECT': 263, 'EUROPE': 264,
                         'ACTION': 265, 'TRUST': 266, 'MOVING': 267, 'QUITE': 268, 'SAYING': 269, 'PRICE': 270, 'FORMER': 271, 'THREE': 272,
                         'BEFORE': 273, 'THEIR': 274, 'GROUP': 275, 'LEADERS': 276, 'EDUCATION': 277, 'BELIEVE': 278, 'HEARD': 279,
                         'IMMIGRATION': 280, 'CAMPAIGN': 281, 'LIKELY': 282, 'THESE': 283, 'CONSERVATIVE': 284, 'CHIEF': 285, 'SPEECH': 286,
                         'PEOPLE': 287, 'PARTS': 288, 'LONDON': 289, 'THERE': 290, 'HUNDREDS': 291, 'OFTEN': 292, 'COUPLE': 293, 'GAMES': 294,
                         'PROBABLY': 295, 'WHOLE': 296, 'GIVING': 297, 'CASES': 298, 'FAMILY': 299, 'MONEY': 300, 'BEING': 301, 'BANKS': 302,
                         'FOCUS': 303, 'TOGETHER': 304, 'GERMANY': 305, 'ACTUALLY': 306, 'CONCERNS': 307, 'VOTERS': 308, 'WOMEN': 309, 'STATE': 310,
                         'PROBLEMS': 311, 'TODAY': 312, 'SECOND': 313, 'INFLATION': 314, 'HAPPENED': 315, 'TRIAL': 316, 'SCHOOL': 317, 'DEATH': 318,
                         'TONIGHT': 319, 'MEMBER': 320, 'FINAL': 321, 'ATTACK': 322, 'OUTSIDE': 323, 'THOUSANDS': 324, 'EVERY': 325, 'COMING': 326,
                         'BRING': 327, 'GROWING': 328, 'ACCUSED': 329, 'ISSUES': 330, 'POLITICAL': 331, 'OFFICERS': 332, 'JAMES': 333, 'YEARS': 334,
                         'PARLIAMENT': 335, 'WESTMINSTER': 336, 'EASTERN': 337, 'MAYBE': 338, 'HOUSING': 339, 'MEMBERS': 340, 'HIGHER': 341,
                         'MESSAGE': 342, 'OBAMA': 343, 'STAGE': 344, 'CENTRAL': 345, 'EXTRA': 346, 'PARTY': 347, 'THING': 348, 'ABUSE': 349,
                         'POSSIBLE': 350, 'LEGAL': 351, 'INVESTMENT': 352, 'ECONOMIC': 353, 'POWERS': 354, 'WHERE': 355, 'DECISION': 356,
                         'SOUTHERN': 357, 'GIVEN': 358, 'REPORT': 359, 'FURTHER': 360, 'MAJORITY': 361, 'WITHIN': 362, 'COURT': 363, 'SECURITY': 364,
                         'CLAIMS': 365, 'WELFARE': 366, 'EARLY': 367, 'MILITARY': 368, 'INTEREST': 369, 'MIGHT': 370, 'CALLED': 371, 'PHONE': 372,
                         'WESTERN': 373, 'INDEPENDENT': 374, 'LIVING': 375, 'RIGHT': 376, 'SPENDING': 377, 'PRETTY': 378, 'BEHIND': 379,
                         'ALREADY': 380, 'ATTACKS': 381, 'MISSING': 382, 'BRITISH': 383, 'SIDES': 384, 'POLICY': 385, 'YOUNG': 386, 'BENEFITS': 387,
                         'START': 388, 'PRESIDENT': 389, 'SINCE': 390, 'CHARGES': 391, 'SENIOR': 392, 'SOCIAL': 393, 'GREECE': 394, 'LOCAL': 395,
                         'YESTERDAY': 396, 'AMONG': 397, 'CHILD': 398, 'TALKS': 399, 'MAJOR': 400, 'WHETHER': 401, 'FIGHTING': 402, 'GROWTH': 403,
                         'PROVIDE': 404, 'SEVEN': 405, 'ANSWER': 406, 'UNTIL': 407, 'ACROSS': 408, 'MIGRANTS': 409, 'REPORTS': 410, 'FOLLOWING': 411,
                         'PERIOD': 412, 'SUNDAY': 413, 'BILLION': 414, 'WINDS': 415, 'COUNTRIES': 416, 'SHOULD': 417, 'SPEAKING': 418, 'REALLY': 419,
                         'WANTED': 420, 'SPECIAL': 421, 'IMPORTANT': 422, 'ITSELF': 423, 'USING': 424, 'WORKERS': 425, 'ACCORDING': 426,
                         'ENGLAND': 427, 'HISTORY': 428, 'BORDER': 429, 'CHALLENGE': 430, 'BROUGHT': 431, 'SIGNIFICANT': 432, 'ALLEGATIONS': 433,
                         'NEEDS': 434, 'RESULT': 435, 'ALLOW': 436, 'MOMENT': 437, 'PRESSURE': 438, 'GLOBAL': 439, 'THOUGHT': 440, 'SEVERAL': 441,
                         'QUESTION': 442, 'CLEAR': 443, 'HAPPEN': 444, 'PERSON': 445, 'PRIVATE': 446, 'POSITION': 447, 'ALLOWED': 448, 'RULES': 449,
                         'SOMEONE': 450, 'SERIES': 451, 'THEMSELVES': 452, 'ALWAYS': 453, 'CONTINUE': 454, 'MONTHS': 455, 'INFORMATION': 456,
                         'THREAT': 457, 'SOCIETY': 458, 'STRONG': 459, 'OFFICIALS': 460, 'EUROPEAN': 461, 'FRIDAY': 462, 'CRISIS': 463,
                         'GOVERNMENT': 464, 'COMMUNITY': 465, 'AFFECTED': 466, 'MILLIONS': 467, 'ABOUT': 468, 'DEFICIT': 469, 'IMPACT': 470,
                         'ACCESS': 471, 'BUSINESS': 472, 'THROUGH': 473, 'DURING': 474, 'EDITOR': 475, 'FORCES': 476, 'RECORD': 477, 'LEAVE': 478,
                         'BLACK': 479, 'LEADERSHIP': 480, 'TIMES': 481, 'UNITED': 482, 'WORST': 483, 'LOOKING': 484, 'THINGS': 485, 'ALMOST': 486,
                         'SPENT': 487, 'HUMAN': 488, 'SERVICES': 489, 'KILLED': 490, 'DIFFERENCE': 491, 'DEGREES': 492, 'AGREEMENT': 493,
                         'TRADE': 494, 'AROUND': 495, 'KNOWN': 496, 'NUMBER': 497, 'PLACES': 498, 'AREAS': 499}

if __name__ == "__main__":

    for key, value in args.items():
        print(str(key) + " : " + str(value))
