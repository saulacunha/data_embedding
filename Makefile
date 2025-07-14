.PHONY: build up down restart logs clean test shell help

# Construye las imágenes Docker
build:
	docker-compose build

# Levanta los contenedores en segundo plano
up:
	docker-compose up -d

# Detiene y elimina los contenedores, redes y volúmenes asociados
down:
	docker-compose down --volumes --remove-orphans

# Reinicia los contenedores (down + up)
restart: down up

# Muestra los logs en tiempo real
logs:
	docker-compose logs -f

# Limpia contenedores detenidos, imágenes no usadas, volúmenes y redes huérfanos
clean:
	docker system prune -af --volumes

# Ejecuta pruebas dentro del contenedor principal (ajusta el servicio y comando según tu proyecto)
test:
	docker-compose run --rm app pytest

# Abre una shell interactiva en el contenedor principal (ajusta el nombre del servicio si es necesario)
shell:
	docker-compose exec app sh

# Muestra esta ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  build    - Construye las imágenes Docker"
	@echo "  up       - Levanta los contenedores en segundo plano"
	@echo "  down     - Detiene y elimina los contenedores, redes y volúmenes"
	@echo "  restart  - Reinicia los contenedores"
	@echo "  logs     - Muestra los logs en tiempo real"
	@echo "  clean    - Limpia contenedores, imágenes, volúmenes y redes no usados"
	@echo "  test     - Ejecuta pruebas dentro del contenedor principal"
	@echo "  shell    - Abre una shell interactiva en el contenedor principal"
	@echo "  help     - Muestra esta ayuda"