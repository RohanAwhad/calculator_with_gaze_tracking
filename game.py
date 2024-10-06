import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
HOVER_COLOR: tuple[int, int, int] = (200, 200, 200)  # Light gray for hover effect

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h
WIDTH, HEIGHT = int(SCREEN_WIDTH * 0.8), int(SCREEN_HEIGHT * 0.8)
BUTTON_WIDTH, BUTTON_HEIGHT = WIDTH // 6, HEIGHT // 8
BUTTON_MARGIN = 20

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('BODMAS Calculator')

# Fonts
font = pygame.font.Font(None, 72)

# Calculator state
expression: str = ""
result: str | float | None = None

def draw_button(text: str, x: int, y: int, hover: bool = False) -> pygame.Rect:
    """Draw a button with the specified label at (x, y)."""
    button_rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
    if hover:
        pygame.draw.rect(screen, HOVER_COLOR, button_rect)
        pygame.draw.rect(screen, BLACK, button_rect, 3)  # Draw border
    else:
        pygame.draw.rect(screen, WHITE, button_rect)
    label = font.render(text, True, BLACK)
    label_rect = label.get_rect(center=button_rect.center)
    screen.blit(label, label_rect)
    return button_rect

def evaluate_expression(expr: str) -> str | float:
    """Evaluate the expression using BODMAS rules."""
    try:
        return eval(expr)
    except Exception:
        return "Error"

def main() -> None:
    global expression, result
    button_texts: list[tuple[str, str, str, str, str]] = [
        ('7', '8', '9', '/', 'C'),
        ('4', '5', '6', '*', '←'),
        ('1', '2', '3', '-', '('),
        ('0', '.', '=', '+', ')')
    ]

    # Main loop
    while True:
        screen.fill(BLACK)

        mouse_pos = pygame.mouse.get_pos()

        # Draw buttons
        buttons: list[list[tuple[pygame.Rect, str]]] = []
        for i, row in enumerate(button_texts):
            button_row: list[tuple[pygame.Rect, str]] = []
            for j, text in enumerate(row):
                x = BUTTON_MARGIN + j * (BUTTON_WIDTH + BUTTON_MARGIN)
                y = BUTTON_MARGIN + i * (BUTTON_HEIGHT + BUTTON_MARGIN) + 200
                button_rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
                hover = button_rect.collidepoint(mouse_pos)
                drawn_rect = draw_button(text, x, y, hover)
                button_row.append((drawn_rect, text))
            buttons.append(button_row)

        # Draw expression
        expression_label = font.render(expression, True, WHITE)
        screen.blit(expression_label, (10, 50))

        # Draw result
        if result is not None:
            result_label = font.render(f"= {result}", True, WHITE)
            screen.blit(result_label, (10, 150))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for row in buttons:
                    for button_rect, text in row:
                        if button_rect.collidepoint(event.pos):
                            if text == '=':
                                result = evaluate_expression(expression)
                            elif text == 'C':
                                expression = ""
                                result = None
                            elif text == '←':
                                expression = expression[:-1]
                            else:
                                expression += text

        pygame.display.flip()

if __name__ == "__main__":
    main()
