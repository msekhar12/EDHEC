import unittest
import edhec_risk_kit as erk
import pandas as pd
#pd.options.display.float_format = '{:.10f}'.format

returns = None
drawdown_result = None


class TestEdhecRiskKit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create a returns DataFrame, which will be used in various test cases
        This function will execute only once before the beginning the run of test cases
        '''
        global returns
        global drawdown_result
        returns = pd.DataFrame({'SmallCap': [-0.0145, 0.0512, 0.0093, -0.0484, -0.0078],
                                'LargeCap': [0.0329, 0.0370, 0.0067, -0.0243, 0.0270]})

        drawdown_result = pd.DataFrame({'Returns': [-0.0145, 0.0512, 0.0093, -0.0484, -0.0078],
                                        'Wealth_Index': [985.5, 1035.9576, 1045.59200568,  994.98535261, 987.22446685],
                                        'Previous_Peak': [985.5, 1035.9576, 1045.59200568, 1045.59200568, 1045.59200568],
                                        'Drawdown': [0.,  0.,  0., -0.0484, -0.05582248]})

    def test_drawdowns(self):
        small_cap = erk.drawdown(returns['SmallCap'])
        pd.testing.assert_frame_equal(drawdown_result, small_cap)


if __name__ == '__main__':
    unittest.main()
