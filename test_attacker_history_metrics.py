import unittest
from attacker_history_metrics import possession_entries, regulation_minutes, uncertainty, risk_adjusted_rate, normalize, team_key

def event(pid, team, kind='Pass', period=1, touch=True, minute=0):
    return {'playerId':pid,'teamId':team,'type':{'displayName':kind},
            'period':{'value':period},'isTouch':touch,'minute':minute}

class MetricTests(unittest.TestCase):
    def test_possession_entries_are_not_touches(self):
        stream = [event(1,10),event(1,10,'TakeOn'),event(99,10,'SubstitutionOn',touch=False),
                  event(2,10),event(3,20,'BallRecovery'),event(1,10),event(1,10,period=2)]
        self.assertEqual(dict(possession_entries(stream)),{1:3,2:1,3:1})

    def test_minutes_exclude_unused_bench_and_stop_at_red(self):
        m={'home':{'players':[{'playerId':1,'isFirstEleven':True},{'playerId':2},{'playerId':3}]},
           'away':{'players':[{'playerId':4,'isFirstEleven':True}]},
           'events':[event(1,10,'SubstitutionOff',period=2,minute=60),event(2,10,'SubstitutionOn',period=2,minute=60),
                     dict(event(4,20,'Card',period=2,minute=75),qualifiers=[{'type':{'displayName':'Red'}}])]}
        self.assertEqual(regulation_minutes(m),{1:60,2:30,3:0,4:75})

    def test_no_false_certainty_with_one_short_appearance(self):
        stats=uncertainty([0.1],[5])
        self.assertAlmostEqual(stats['rate_per90'],1.8)
        self.assertIsNone(stats['se_per90'])
        self.assertIsNone(risk_adjusted_rate(1.8,5,.2,None,None)['risk_adjusted_rate'])

    def test_ratio_rate_and_volatility_are_separate(self):
        stats=uncertainty([0,.2,0,.2,0,.2],[90]*6)
        self.assertAlmostEqual(stats['rate_per90'],.1)
        self.assertGreater(stats['weighted_sd_per90'],stats['se_per90'])
        base=risk_adjusted_rate(.3,900,.1,.02,.1)
        conservative=risk_adjusted_rate(.3,900,.1,.02,.1,volatility_weight=1)
        self.assertLess(conservative['risk_adjusted_rate'],base['risk_adjusted_rate'])

    def test_explicit_aliases_preserve_other_names(self):
        self.assertEqual(normalize('Yéremy Pino'),normalize('Yeremi Pino'))
        self.assertEqual(team_key('Brighton and Hove Albion'),team_key('Brighton'))
        self.assertNotEqual(normalize('Thierno Barry'),normalize('Thierno Baldé'))
