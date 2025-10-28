FILE_PATH = "vacinados.csv"
TARGET_COLUMN = "Tipo de Dose"  
SEED = 42
AMOSTRA_LIMITE = 50000

EXPECTED_COLUMNS = [
    'faixa_etaria','idade','sexo','raca_cor','municipio','grupo','categoria',
    'lote','vacina_fabricante','descricao_dose','cnes','sistema_origem','data_vacinacao'
]