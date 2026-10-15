import unittest
import numpy as np
from sgtr_demo import adjacency, customer_mean_abs_change, derivatives, loss, sigmoid, stable_gate


class TestSGTR(unittest.TestCase):
    def setUp(self):
        self.f = np.array([-.8, .4, 1.1, -.1])
        self.y = np.array([0., 0., 1., 1.])
        self.e = np.array([[0, 1], [2, 3]])
        self.w = np.array([1., 0.])

    def test_gradient_and_hessian_diagonal_finite_difference(self):
        grad, diag = derivatives(self.f, self.y, self.e, self.w)
        eps = 1e-5
        for j in range(len(self.f)):
            step = np.eye(len(self.f))[j] * eps
            fp, fm = self.f + step, self.f - step
            numerical_grad = (loss(fp, self.y, self.e, self.w) - loss(fm, self.y, self.e, self.w))/(2*eps)
            numerical_h = (derivatives(fp, self.y, self.e, self.w)[0][j] - derivatives(fm, self.y, self.e, self.w)[0][j])/(2*eps)
            self.assertAlmostEqual(grad[j], numerical_grad, places=8)
            self.assertAlmostEqual(diag[j], numerical_h, places=8)

    def test_lambda_zero_is_row_bce(self):
        g, h = derivatives(self.f, self.y, self.e, self.w, 0)
        p = sigmoid(self.f)
        np.testing.assert_array_equal(g, p-self.y)
        np.testing.assert_array_equal(h, p*(1-p))

    def test_gate_boundaries_and_missing(self):
        np.testing.assert_array_equal(stable_gate([0, .25, .250001, .9, 1, np.nan]), [1,1,0,0,0,0])

    def test_nonstable_rows_still_receive_classification_gradients(self):
        g, _ = derivatives(self.f, self.y, self.e, self.w)
        np.testing.assert_array_equal(g[2:], sigmoid(self.f)[2:]-self.y[2:])
        self.assertTrue(np.all(g[2:] != 0))

    def test_no_cross_customer_or_gap_edges(self):
        edges = adjacency(['A','B','A','A','B'], ['2020-01','2020-02','2020-02','2020-04','2020-03'])
        np.testing.assert_array_equal(edges, [[0,2],[1,4]])

    def test_year_boundary_and_unordered_rows(self):
        np.testing.assert_array_equal(adjacency(['A','A'], ['2021-01','2020-12']), [[1,0]])

    def test_duplicate_customer_month_rejected(self):
        with self.assertRaises(ValueError):
            adjacency(['A','A'], ['2020-01','2020-01'])

    def test_train_partition_edges_only(self):
        ids = np.array(['A','A','B','B'])
        months = np.array(['2020-01','2020-02','2020-01','2020-02'])
        selected = ids == 'A'
        e = adjacency(ids[selected], months[selected])
        self.assertEqual(len(e), 1)
        self.assertTrue(np.all(ids[selected][e] == 'A'))

    def test_sigmoid_extremes_are_finite(self):
        with np.errstate(over='raise', invalid='raise'):
            p = sigmoid([-10000, 0, 10000])
        np.testing.assert_array_equal(p, [0, .5, 1])

    def test_customer_equal_not_pair_equal(self):
        # A contributes two changes of 1, B one change of 3.
        m = customer_mean_abs_change([0,1,2,0,3], [[0,1],[1,2],[3,4]], ['A','A','B'], [True]*3)
        self.assertEqual(m, 2.)

    def test_empty_edges(self):
        g, h = derivatives(self.f, self.y, [], [])
        np.testing.assert_array_equal(g, sigmoid(self.f)-self.y)
        self.assertTrue(np.all(h >= 0))

    def test_invalid_edges_rejected(self):
        for e in ([[0,0]], [[0,7]], [[0,1],[1,0]]):
            with self.assertRaises(ValueError):
                loss(self.f, self.y, e, np.ones(len(e)))


if __name__ == '__main__':
    unittest.main()
