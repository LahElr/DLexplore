import torch


class test:
    def __init__(self,tester,testee,eps=1e-10):
        self.tester = tester
        self.testee = testee
        self.eps = eps

    def test(self,data, input_is_param_dict=False):
        if input_is_param_dict:
            gt = self.tester(**data)
            ans = self.tester(**data)
        else:
            gt = self.tester(data)
            ans = self.tester(data)
        return torch.all(gt-ans<self.eps).item()
    
    def print_test(self,data, input_is_param_dict=False):
        answer = self.test(data, input_is_param_dict)
        if answer:
            print(f'Test shows {self.tester.__name__} and {self.testee.__name__} are equal given the specified input.')
        else:
            print(f'test failed')

    def batch_test(self,cases,input_is_param_dict=False):
        results = []
        for case in cases:
            results.append(self.test(case,input_is_param_dict))
        if all(results):
            return True
        else:
            return results
    
    def print_batch_test(self,cases,input_is_param_dict=False):
        results = self.batch_test(cases,input_is_param_dict)
        if not isinstance(results,list):
            print(f'Test shows {self.tester.__name__} and {self.testee.__name__} are equal given the specified inputs.')
        else:
            print('test failed for the following cases:')
            for i,_ in enumerate(results):
                if not _:
                    print(f'{i}-th')
    