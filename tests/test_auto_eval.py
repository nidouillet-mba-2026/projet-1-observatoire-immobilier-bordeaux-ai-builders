"""
Evaluation automatique - Projet 1 : Observatoire Immobilier Toulonnais
=======================================================================
Ce fichier est execute par le CI a chaque push.
NE MODIFIEZ PAS ce fichier.

Bareme automatise : 55 points / 100
Les 45 points restants sont evalues en soutenance.
"""

import ast
import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# STATS FROM SCRATCH (15 pts)
# ===========================================================================
class TestStatsScratch:
    """Fonctions statistiques implementees from scratch - Grus ch.5 (15 pts)."""

    @pytest.fixture
    def stats(self):
        sys.path.insert(0, PROJECT_ROOT)
        try:
            import analysis.stats as m
            return m
        except ImportError:
            pytest.fail(
                "analysis/stats.py introuvable. "
                "Creez ce fichier avec vos fonctions from scratch."
            )
        finally:
            sys.path.pop(0)

    def test_mean_correct(self, stats):
        """[3 pts] mean() calcule la moyenne correctement."""
        assert stats.mean([1, 2, 3, 4, 5]) == 3.0
        assert stats.mean([10, 20]) == 15.0

    def test_variance_correct(self, stats):
        """[4 pts] variance() calcule la variance correctement."""
        # variance de [2,4,4,4,5,5,7,9] = 4.0
        result = stats.variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(result - 4.0) < 0.1, f"Variance attendue ~4.0, obtenue {result}"

    def test_correlation_correct(self, stats):
        """[4 pts] correlation() retourne 1.0 pour deux series identiques."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = stats.correlation(xs, xs)
        assert abs(result - 1.0) < 0.01, f"Correlation attendue 1.0, obtenue {result}"

    def test_no_numpy_in_stats(self, stats):
        """[4 pts] analysis/stats.py n'importe pas numpy (from scratch)."""
        path = os.path.join(PROJECT_ROOT, "analysis", "stats.py")
        if not os.path.exists(path):
            pytest.skip("analysis/stats.py absent")
        with open(path) as f:
            source = f.read()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any("numpy" in name for name in imported), (
            "analysis/stats.py importe numpy. "
            "Implementez les fonctions with pure Python, sans numpy."
        )


# ===========================================================================
# REGRESSION FROM SCRATCH (10 pts)
# ===========================================================================
class TestRegressionScratch:
    """Regression lineaire implementee from scratch - Grus ch.14 (10 pts)."""

    @pytest.fixture
    def reg(self):
        sys.path.insert(0, PROJECT_ROOT)
        try:
            import analysis.regression as m
            return m
        except ImportError:
            pytest.fail(
                "analysis/regression.py introuvable. "
                "Creez ce fichier avec least_squares_fit() et r_squared()."
            )
        finally:
            sys.path.pop(0)

    def test_least_squares_fit(self, reg):
        """[5 pts] least_squares_fit() trouve alpha et beta corrects pour y = 2x + 1."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        alpha, beta = reg.least_squares_fit(x, y)
        assert abs(beta - 2.0) < 0.01, f"Beta attendu ~2.0, obtenu {beta}"
        assert abs(alpha - 1.0) < 0.01, f"Alpha attendu ~1.0, obtenu {alpha}"

    def test_r_squared_perfect_fit(self, reg):
        """[3 pts] r_squared() retourne 1.0 pour une relation lineaire parfaite."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 7.0, 9.0, 11.0]
        alpha, beta = reg.least_squares_fit(x, y)
        r2 = reg.r_squared(alpha, beta, x, y)
        assert abs(r2 - 1.0) < 0.01, f"R² attendu 1.0, obtenu {r2}"

    def test_no_sklearn_in_regression(self, reg):
        """[2 pts] analysis/regression.py n'importe pas sklearn."""
        path = os.path.join(PROJECT_ROOT, "analysis", "regression.py")
        if not os.path.exists(path):
            pytest.skip("analysis/regression.py absent")
        with open(path) as f:
            source = f.read()
        assert "sklearn" not in source, (
            "analysis/regression.py importe sklearn. "
            "Implementez la regression from scratch (voir Grus ch.14)."
        )


# ===========================================================================
# DONNEES REELLES (15 pts)
# ===========================================================================
class TestDonnees:
    """Donnees reelles collectees (15 pts)."""

    def _find_dvf(self):
        for name in ["dvf_toulon.csv", "dvf.csv", "dvf_83137.csv"]:
            p = os.path.join(PROJECT_ROOT, "data", name)
            if os.path.exists(p):
                return p
        return None

    def _find_annonces(self):
        for name in ["annonces.csv", "annonces_actuelles.csv", "annonces_toulon.csv"]:
            p = os.path.join(PROJECT_ROOT, "data", name)
            if os.path.exists(p):
                return p
        return None

    def test_dvf_exists(self):
        """[5 pts] Fichier DVF present dans data/."""
        assert self._find_dvf() is not None, (
            "Aucun fichier DVF trouve dans data/. "
            "Telechargez depuis https://files.data.gouv.fr/geo-dvf/latest/csv/83/"
        )

    def test_dvf_min_rows(self):
        """[5 pts] DVF contient au moins 500 transactions."""
        path = self._find_dvf()
        if path is None:
            pytest.skip("DVF absent")
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) >= 500, (
            f"DVF contient {len(df)} lignes, minimum 500 requis."
        )

    def test_annonces_exists(self):
        """[5 pts] Fichier annonces reelles present dans data/."""
        assert self._find_annonces() is not None, (
            "Aucun fichier annonces trouve dans data/. "
            "Collectez des annonces reelles (GumLoop ou scraping)."
        )


# ===========================================================================
# APPLICATION ET DEPLOIEMENT (10 pts)
# ===========================================================================
class TestApplication:
    """Application Streamlit deployee (10 pts)."""

    def test_app_file_exists(self):
        """[4 pts] Fichier de l'application Streamlit present."""
        candidates = [
            os.path.join(PROJECT_ROOT, "app", "streamlit_app.py"),
            os.path.join(PROJECT_ROOT, "app.py"),
            os.path.join(PROJECT_ROOT, "streamlit_app.py"),
        ]
        assert any(os.path.exists(p) for p in candidates), (
            "Fichier Streamlit introuvable. "
            "Attendu : app/streamlit_app.py ou app.py a la racine."
        )

    def test_readme_has_deployed_url(self):
        """[6 pts] README contient une URL de deploiement public."""
        readme = os.path.join(PROJECT_ROOT, "README.md")
        assert os.path.exists(readme), "README.md introuvable."
        with open(readme) as f:
            content = f.read()
        patterns = [
            r"https://[a-zA-Z0-9\-]+\.streamlit\.app",
            r"https://[a-zA-Z0-9\-]+\.railway\.app",
            r"https://[a-zA-Z0-9\-]+\.onrender\.com",
            r"https://[a-zA-Z0-9\-]+\.fly\.dev",
            r"https://huggingface\.co/spaces/",
        ]
        assert any(re.search(p, content) for p in patterns), (
            "README.md ne contient pas d'URL de deploiement. "
            "Deployez votre app (Streamlit Cloud, Railway, Render...) "
            "et ajoutez le lien dans le README."
        )


# ===========================================================================
# TESTS ETUDIANTS (5 pts)
# ===========================================================================
class TestStructure:
    """Tests ecrits par les etudiants (5 pts)."""

    def test_student_tests_exist(self):
        """[5 pts] Au moins 3 tests unitaires ecrits par l'equipe."""
        total = 0
        for name in ["test_stats.py", "test_regression.py", "test_scoring.py"]:
            p = os.path.join(PROJECT_ROOT, "tests", name)
            if os.path.exists(p):
                with open(p) as f:
                    total += len(re.findall(r"def (test_\w+)", f.read()))
        assert total >= 3, (
            f"Seulement {total} test(s) trouves dans tests/. "
            "Ecrivez au moins 3 tests pour vos fonctions from scratch."
        )
