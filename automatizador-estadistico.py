import numpy as np
import matplotlib.pyplot as plt  # Para graficar


def get_data():
    """
    Solicita los datos al usuario y devuelve una lista de números.
    """
    while True:
        try:
            data = input(
                "Ingresa los datos separados por comas (ej. 4, 8, 6, 5, 7): ")
            # Filtramos valores vacíos antes de convertirlos
            data = [float(i.strip()) for i in data.split(',') if i.strip()]
            if len(data) == 0:  # Validamos que no esté vacío
                raise ValueError(
                    "No ingresaste datos válidos. Intenta de nuevo.")
            return data
        except ValueError as e:
            print(f"Error: {e}")


def calculate_covariance(data1, data2):
    """
    Calcula la covarianza entre dos conjuntos de datos.
    """
    return np.cov(data1, data2)[0][1]


def calculate_correlation_coefficient(data1, data2):
    """
    Calcula el coeficiente de correlación entre dos conjuntos de datos.
    """
    return np.corrcoef(data1, data2)[0][1]


def plot_scatter(data1, data2):
    """
    Crea un gráfico de dispersión y lo muestra en pantalla.
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(data1, data2, alpha=0.7, color='red')
    plt.title("Gráfico de dispersión")
    plt.xlabel("Datos 1")
    plt.ylabel("Datos 2")
    plt.grid(True)
    plt.show()  # Mostrar el grafico en pantalla


def plot_data_with_regression_line(x_data, y_data):
    """
    Crea un gráfico de dispersión con una línea de regresión.
    """
    # Realizamos la regresión lineal
    # Regresión lineal: m es la pendiente y b la intersección
    m, b = np.polyfit(x_data, y_data, 1)
    plt.figure(figsize=(8, 4))
    plt.scatter(x_data, y_data, color='blue',
                label='Datos')  # Los puntos de datos
    plt.plot(x_data, m*np.array(x_data) + b, color='red',
             label='Línea de regresión')  # Línea de regresión
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.show()  # Mostrar el gráfico en pantalla


def compare_two_sets():
    """
    Permite al usuario comparar dos conjuntos de datos y calcular la covarianza y el coeficiente de correlación.
    """
    print("Comparando dos conjuntos de datos...")
    data1 = get_data()
    print(f"Primer conjunto de datos: {data1}")

    data2 = get_data()
    print(f"Segundo conjunto de datos: {data2}")

    # Calcular la covarianza y el coeficiente de correlación
    covariance = calculate_covariance(data1, data2)
    correlation = calculate_correlation_coefficient(data1, data2)

    print(f"\nCovarianza entre los conjuntos de datos: {covariance:.2f}")
    print(f"Coeficiente de correlación entre los conjuntos de datos: {
          correlation:.2f}")

    # Mostrar el gráfico de dispersión
    plot_scatter(data1, data2)


def main():
    """
    Función principal del programa de estadísticas.
    """
    while True:
        print("\nBienvenido al programa de estadísticas.")
        # Opción para calcular estadísticas de un solo conjunto de datos
        option = input(
            "¿Quieres trabajar con un solo conjunto de datos o comparar dos? (1 para uno, 2 para comparar): ")

        if option == "1":
            data = get_data()
            print(f"Datos ingresados: {data}")
            # Graficar datos con regresión
            plot_data_with_regression_line(range(1, len(data)+1), data)

        elif option == "2":
            compare_two_sets()  # Comparar dos conjuntos de datos

        else:
            print("Opción no válida. Por favor, elige 1 o 2.")
            continue  # Vuelve al inicio del bucle

        # Preguntar al usuario si desea reiniciar
        restart = input(
            "\n¿Quieres realizar otro análisis? (s para sí, cualquier otra tecla para salir): ").lower()
        if restart != 's':
            print("Gracias por usar el programa de estadísticas. ¡Adiós!")
            break  # Sale del bucle y termina el programa


if __name__ == "__main__":
    main()
