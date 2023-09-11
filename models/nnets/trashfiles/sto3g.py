#!/usr/bin/python3

import re
import numpy as np

class STO3GBasis():

    def __init__(self, sto3g_data_file_path):
        self.sto3g_data_file_path = sto3g_data_file_path
        self.coeff_dict = self.create_coeffs_dict(sto3g_data_file_path)


    def gto(self, x, y, z, alpha, orbital):
        if "-" in orbital:
            return self.gto(x, y, z, alpha, "xx") - self.gto(x, y, z, alpha, "yy")
        else:
            i = len([s for s in orbital if s == "x"])
            j = len([s for s in orbital if s == "y"])
            k = len([s for s in orbital if s == "z"])
        prefac1 = (2 * alpha / np.pi) ** (3/4)
        prefac2 = 1
        for n in [i, j, k]:
            prefac2 *= (8 * alpha) ** n
            prefac2 *= np.math.factorial(n) / np.math.factorial(2*n)
        prefac2 = prefac2 ** 0.5
        prefac = prefac1 * prefac2
        return prefac * x ** i * y ** j * z ** k * np.exp(-alpha * (x ** 2 + y ** 2 + z ** 2))
    

    def sto(self, x, y, z, atom, orbital):
        alphas = self.coeff_dict[atom][orbital[:2]]["alphas"]
        cs = self.coeff_dict[atom][orbital[:2]]["cs"]
        gtos = []
        for i in range(len(alphas)):
            orbital = orbital.replace("z^2", "zz")
            orbital = orbital.replace("x^2", "xx")
            orbital = orbital.replace("y^2", "yy")
            alpha = float(alphas[i])
            c = float(cs[i])
            gto_i = c * self.gto(x, y, z, alpha, orbital)
            gtos.append(gto_i)
        return sum(gtos)  

    def create_coeffs_dict(self, file_path):

        def analyze_schema(schema):
            pattern = "([\d])s"
            if "p" in schema:
                pattern += ",([\d])p"
            if "d" in schema:
                pattern += ",([\d])d"
            match = re.search(pattern, schema)
            s = match.group(1)
            p = match.group(2) if "p" in schema else 0
            d = match.group(3) if "d" in schema else 0
            return {"s": int(s), "p": int(p), "d": int(d)}

        def truncate_row(row):
            while (row.replace("  ", " ") != row):
                row = row.replace("  ", " ")
            return row

        raw_data = open(file_path, "r").read()
        raw_data = raw_data.split("#BASIS SET: ")[1:]
        pattern = r"\n(\w+)\s"
        coeffs_dict = {}
        for row in raw_data:
            match = re.search(pattern, row)
            element = match.group(1)
            row_data = row.split("\n")[:-1]
            schema = row_data[0].split("-> ")[1]
            schema = analyze_schema(schema)
            row_data = row_data[1:]
            for i in range(len(row_data)):
                if (i + 1) % 4 == 0:
                    row_data[i] += "||"
            row_data = "".join(row_data)
            row_data = truncate_row(row_data)
            row_data = row_data.split("||")[:-1]
            s = schema["s"]
            d = schema["d"]
            s_orbital = ["1s"]
            sp_orbitals = ["{}sp".format(i) for i in range(2, s + 1)]
            d_orbitals = ["{}d".format(i) for i in range(1, d + 1)]
            orbitals = s_orbital + sp_orbitals + d_orbitals
            coeffs_dict[element] = {}
            for i in range(len(orbitals)):
                orb = orbitals[i]
                coeffs = row_data[i]
                coeffs = coeffs.split(" ")[2:]
                coeffs = [c for c in coeffs if len(c) > 0]
                columns = len(coeffs) // 3
                alphas = [coeffs[i] for i in range(len(coeffs)) if i % columns == 0]
                if "sp" in orb:
                    scs = [coeffs[i] for i in range(len(coeffs)) if i % columns == 1]
                    pcs = [coeffs[i] for i in range(len(coeffs)) if i % columns == 2]
                    s_alphas_cs = {"alphas": alphas, "cs": scs}
                    p_alphas_cs = {"alphas": alphas, "cs": pcs}
                    s_orb = orb.replace("p", "")
                    p_orb = orb.replace("s", "")
                    coeffs_dict[element][s_orb] = s_alphas_cs
                    coeffs_dict[element][p_orb] = p_alphas_cs
                else:
                    cs = [coeffs[i] for i in range(len(coeffs)) if i % columns == 1]
                    alphas_cs = {"alphas": alphas, "cs": cs}
                    coeffs_dict[element][orb] = alphas_cs
        return coeffs_dict