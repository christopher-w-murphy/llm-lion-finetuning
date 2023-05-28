from datasets import Dataset, DatasetDict
from pytest import fixture

from src.domain.transform import concatenate_train_test_data


@fixture(scope='session')
def samsum_train_samples() -> Dataset:
    train_samples = [
        {
            "id": "13818513",
            "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)",
            "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
        },
        {
            'id': '13728867',
            'dialogue': 'Olivia: Who are you voting for in this election? \r\nOliver: Liberals as always.\r\nOlivia: Me too!!\r\nOliver: Great',
            'summary': 'Olivia and Olivier are voting for liberals in this election. '
        }
    ]
    return Dataset.from_list(train_samples)


@fixture(scope='session')
def samsum_test_samples() -> Dataset:
    test_samples = [
        {
            'id': '13862856',
            'dialogue': "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye",
            'summary': "Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry."
        },
        {
            'id': '13729565',
            'dialogue': "Eric: MACHINE!\r\nRob: That's so gr8!\r\nEric: I know! And shows how Americans see Russian ;)\r\nRob: And it's really funny!\r\nEric: I know! I especially like the train part!\r\nRob: Hahaha! No one talks to the machine like that!\r\nEric: Is this his only stand-up?\r\nRob: Idk. I'll check.\r\nEric: Sure.\r\nRob: Turns out no! There are some of his stand-ups on youtube.\r\nEric: Gr8! I'll watch them now!\r\nRob: Me too!\r\nEric: MACHINE!\r\nRob: MACHINE!\r\nEric: TTYL?\r\nRob: Sure :)",
            'summary': 'Eric and Rob are going to watch a stand-up on youtube.'
        }
    ]
    return Dataset.from_list(test_samples)


@fixture(scope='session')
def samsum_samples(samsum_train_samples, samsum_test_samples) -> DatasetDict:
    return DatasetDict({'train': samsum_train_samples, 'test': samsum_test_samples})


def test_concatenate_train_test_data(samsum_samples: DatasetDict):
    dataset = concatenate_train_test_data(samsum_samples)
    assert dataset.num_rows == 2 + 2
    assert dataset.num_columns == 3
