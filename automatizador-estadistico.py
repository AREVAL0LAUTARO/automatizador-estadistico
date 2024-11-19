import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_data():
    """
    Solicita los datos al usuario y devuelve una lista de números en el rango 0-1,000,000.
    """
    while True:
        try:
            data = input(
                "Ingresa los datos separados por comas (ej. 4, 8, 6, 5, 7): ")
            data = [float(i.strip()) for i in data.split(',') if i.strip()]

            # Validar que los números estén dentro del rango permitido
            if any(num < 0 or num > 1_000_000 for num in data):
                raise ValueError(
                    "Todos los números deben estar entre 0 y 1,000,000.")

            if len(data) == 0:
                raise ValueError(
                    "No ingresaste datos válidos. Intenta de nuevo.")

            return data
        except ValueError as e:
            print(f"Error: {e}")


def calculate_mean(data):
    return np.mean(data)


def calculate_median(data):
    return np.median(data)


def calculate_mode(data):
    """
    Calcula la moda de un conjunto de datos.
    Devuelve "NO HAY MODA" si no existe moda.
    """
    mode = stats.mode(data, keepdims=True)
    if mode.count[0] > 1:
        return mode.mode[0]
    else:
        return "NO HAY MODA"


def calculate_variance(data):
    return np.var(data, ddof=1)


def calculate_standard_deviation(data):
    return np.std(data, ddof=1)


def calculate_coefficient_of_variation(data):
    """
    Calcula el coeficiente de variación (CV) como un porcentaje.
    """
    mean = calculate_mean(data)
    if mean == 0:
        return 0
    std_dev = calculate_standard_deviation(data)
    return (std_dev / mean) * 100


def adjust_data_lengths(data1, data2):
    """
    Ajusta las longitudes de dos conjuntos de datos al mínimo común.
    """
    min_length = min(len(data1), len(data2))
    return data1[:min_length], data2[:min_length]


def calculate_covariance(data1, data2):
    data1, data2 = adjust_data_lengths(data1, data2)
    return np.cov(data1, data2)[0][1]


def calculate_correlation_coefficient(data1, data2):
    data1, data2 = adjust_data_lengths(data1, data2)
    return np.corrcoef(data1, data2)[0][1]


def display_statistics(data, label='Conjunto de datos'):
    """
    Muestra estadísticas básicas de un conjunto de datos.
    """
    print(f"\nEstadísticas para {label}:")
    print(f"Media: {calculate_mean(data):.2f}")
    print(f"Mediana: {calculate_median(data):.2f}")
    print(f"Moda: {calculate_mode(data)}")
    print(f"Varianza: {calculate_variance(data):.2f}")
    print(f"Desviación estándar: {calculate_standard_deviation(data):.2f}")
    print(f"Coeficiente de variación: {
          calculate_coefficient_of_variation(data):.2f}%")


def plot_histogram(data, filename='histogram.png'):
    """
    Crea un histograma y lo guarda como archivo PNG.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=5, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Histograma de los datos")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_scatter(data1, data2, filename='scatter_plot.png'):
    """
    Crea un gráfico de dispersión y lo guarda como archivo PNG.
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(data1, data2, alpha=0.7, color='red')
    plt.title("Gráfico de dispersión")
    plt.xlabel("Datos 1")
    plt.ylabel("Datos 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_data_with_regression_line(x_data, y_data):
    """
    Grafica los datos con una línea de regresión.
    """
    m, b = np.polyfit(x_data, y_data, 1)
    plt.scatter(x_data, y_data, color='blue', label='Datos')
    plt.plot(x_data, m * np.array(x_data) + b,
             color='red', label='Línea de regresión')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.savefig('regression_line_plot.png')
    plt.show()


def compare_two_sets():
    """
    Compara dos conjuntos de datos, mostrando estadísticas y gráficos.
    """
    data1 = get_data()
    display_statistics(data1, "Primer conjunto de datos")

    data2 = get_data()
    display_statistics(data2, "Segundo conjunto de datos")

    # Ajustar longitudes si son diferentes
    if len(data1) != len(data2):
        print("\nLos conjuntos de datos tienen longitudes diferentes. Se ajustarán automáticamente.")
        data1, data2 = adjust_data_lengths(data1, data2)

    print("\nComparación entre los dos conjuntos de datos:")
    print(f"Covarianza: {calculate_covariance(data1, data2):.2f}")
    print(f"Coeficiente de correlación: {
          calculate_correlation_coefficient(data1, data2):.2f}")

    plot_scatter(data1, data2)
    print("\nSe ha generado el gráfico de dispersión entre los dos conjuntos.")


def main():
    """
    Función principal del programa de estadísticas.
    """
    while True:
        print("\nBienvenido al programa de estadísticas.")
        option = input(
            "¿Quieres trabajar con un solo conjunto de datos o comparar dos? (1 para uno, 2 para comparar): ")

        if option == "1":
            data = get_data()
            display_statistics(data)

            if len(data) > 1:
                plot_option = input(
                    "¿Quieres graficar los datos con regresión lineal? (s para sí, otra tecla para no): ").lower()
                if plot_option == 's':
                    x_data = list(range(1, len(data) + 1))
                    y_data = data
                    plot_data_with_regression_line(x_data, y_data)
        elif option == "2":
            compare_two_sets()
        else:
            print("Opción no válida. Por favor, elige 1 o 2.")
            continue

        restart = input(
            "\n¿Quieres realizar otro análisis? (s para sí, otra tecla para salir): ").lower()
        if restart != 's':
            print("Gracias por usar el programa de estadísticas. ¡Adiós!")
            break


if __name__ == "__main__":
    main()
