// macOS のメニューバー（NSMenu / NSApp.mainMenu）の実装。
// runtime/window.c から呼ばれ、macOS でウィンドウを使うプログラムにだけリンクされる（src/driver.ts）。
//
// - windowMenuBegin 〜 windowMenuCommit で組み立てたメニューの木を NSMenu にして NSApp.mainMenu に据える。
// - SDL_VIDEO_DRIVER が offscreen / dummy のときは本物の画面を出さないので、NSApp のメニューは触らず 0 を返す。
// - 項目のショートカット（keyEquivalent）はメニューの右端に OS の標準の記号（⌘・⌥・⇧・⌃）で表示させるが、
//   キーを押したときの実行は SDL の KeyDown に一本化する（UMMainMenu の performKeyEquivalent: の間だけ
//   UMMenuTarget の validateMenuItem: を NO にする）。これで uikit の文字の入力欄（⌘C / ⌘V / ⌘Z など）や
//   開いている小窓・ドラッグ中のキーの抑止と衝突せず、二重にも発火しない。
//   OS の標準の役割（隠す ⌘H・しまう ⌘M・フルスクリーン ⌃⌘F など）だけは target が NSApp / nil なのでそのまま動く。

#import <AppKit/AppKit.h>
#include <SDL3/SDL.h>
#include <stdint.h>
#include <string.h>

// window.c の修飾キーのビットと同じ。
#define MOD_SHIFT 1
#define MOD_CTRL  2
#define MOD_ALT   4
#define MOD_META  8

// window.um の MenuRole の番号。
enum {
  ROLE_SERVICES = 1,
  ROLE_HIDE = 2,
  ROLE_HIDE_OTHERS = 3,
  ROLE_SHOW_ALL = 4,
  ROLE_MINIMIZE = 5,
  ROLE_ZOOM = 6,
  ROLE_FULLSCREEN = 7,
  ROLE_BRING_ALL_FRONT = 8,
};

static int in_key_equivalent;
static Uint32 menu_event_type;

@interface UMMainMenu : NSMenu
@end

@implementation UMMainMenu
- (BOOL)performKeyEquivalent:(NSEvent *)event {
  in_key_equivalent = 1;
  BOOL r = [super performKeyEquivalent:event];
  in_key_equivalent = 0;
  return r;
}
@end

@interface UMMenuTarget : NSObject <NSMenuItemValidation>
- (void)onMenuItem:(NSMenuItem *)sender;
@end

@implementation UMMenuTarget
- (void)onMenuItem:(NSMenuItem *)sender {
  if (!menu_event_type) return;
  NSString *cmd = sender.representedObject;
  if (![cmd isKindOfClass:[NSString class]] || cmd.length == 0) return;
  const char *utf8 = cmd.UTF8String;
  if (!utf8) return;
  char *copy = SDL_strdup(utf8);
  if (!copy) return;
  SDL_Event e;
  SDL_zero(e);
  e.type = menu_event_type;
  e.user.data1 = copy;
  if (!SDL_PushEvent(&e)) SDL_free(copy);
}

- (BOOL)validateMenuItem:(NSMenuItem *)item {
  if (in_key_equivalent) return NO;
  return item.tag != 0;
}
@end

static UMMenuTarget *menu_target;

// いま組み立て中のメニューバーと、反映済みの ID → NSMenuItem の表。
static UMMainMenu *staged_bar;
static NSMutableArray<NSMenu *> *staged_stack;
static NSMutableDictionary<NSString *, NSMenuItem *> *staged_items;
static NSMenu *staged_services;
static NSMenu *staged_windows;
static NSMutableDictionary<NSString *, NSMenuItem *> *live_items;

int32_t um_mac_menu_has_native(void) {
  const char *drv = SDL_GetCurrentVideoDriver();
  if (!drv) drv = SDL_getenv("SDL_VIDEO_DRIVER");
  if (drv && (strcmp(drv, "offscreen") == 0 || strcmp(drv, "dummy") == 0)) return 0;
  return 1;
}

static void clear_staged_menu(void) {
  staged_bar = nil;
  staged_stack = nil;
  staged_items = nil;
  staged_services = nil;
  staged_windows = nil;
}

void um_mac_menu_begin(void) {
  @autoreleasepool {
    staged_bar = [[UMMainMenu alloc] initWithTitle:@""];
    staged_stack = [NSMutableArray arrayWithObject:staged_bar];
    staged_items = [NSMutableDictionary dictionary];
    staged_services = nil;
    staged_windows = nil;
  }
}

static NSString *ns_str(const char *s) {
  if (!s) return @"";
  NSString *r = [NSString stringWithUTF8String:s];
  return r ? r : @"";
}

// kind: 0 ふつうのトップメニュー、1 アプリのメニュー、2 ウィンドウのメニュー、3 サブメニュー。
void um_mac_menu_begin_menu(int32_t kind, const char *title) {
  @autoreleasepool {
    if (!staged_bar) um_mac_menu_begin();
    NSMenu *parent = (kind == 3 && staged_stack.count > 0) ? staged_stack.lastObject : staged_bar;
    NSString *t = ns_str(title);
    NSMenuItem *holder = [[NSMenuItem alloc] initWithTitle:t action:nil keyEquivalent:@""];
    NSMenu *sub = [[NSMenu alloc] initWithTitle:t];
    holder.submenu = sub;
    [parent addItem:holder];
    if (kind == 2) staged_windows = sub;
    [staged_stack addObject:sub];
  }
}

void um_mac_menu_end(void) {
  @autoreleasepool {
    if (staged_stack.count > 1) [staged_stack removeLastObject];
  }
}

void um_mac_menu_sep(void) {
  @autoreleasepool {
    if (staged_stack.count > 1) [staged_stack.lastObject addItem:[NSMenuItem separatorItem]];
  }
}

// UM のキーの名前（"S"・"Left"・"F1"・"Delete" など）を NSMenuItem の keyEquivalent の文字列にする。
// 大文字の英字をそのまま渡すと Cocoa が勝手に Shift（⇧）を足すので、英字は小文字にして Shift は mask で渡す。
static NSString *key_equiv_of(const char *key, NSEventModifierFlags *mask_io) {
  if (!key || !key[0]) return @"";
  if (key[1] == 0) {
    unsigned char c = (unsigned char)key[0];
    if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
    char buf[2] = { (char)c, 0 };
    return ns_str(buf);
  }
  if ((key[0] == 'F' || key[0] == 'f') && key[1] >= '1' && key[1] <= '9') {
    int n = 0;
    for (int i = 1; key[i]; i++) {
      if (key[i] < '0' || key[i] > '9') { n = 0; break; }
      n = n * 10 + (key[i] - '0');
    }
    if (n >= 1 && n <= 12) {
      unichar ch = (unichar)(NSF1FunctionKey + (n - 1));
      return [NSString stringWithCharacters:&ch length:1];
    }
  }
  unichar special = 0;
  if (SDL_strcasecmp(key, "Enter") == 0 || SDL_strcasecmp(key, "Return") == 0) special = NSCarriageReturnCharacter;
  else if (SDL_strcasecmp(key, "Backspace") == 0) special = NSBackspaceCharacter;
  else if (SDL_strcasecmp(key, "Delete") == 0) special = NSDeleteCharacter;
  else if (SDL_strcasecmp(key, "Tab") == 0) special = NSTabCharacter;
  else if (SDL_strcasecmp(key, "Space") == 0) special = ' ';
  else if (SDL_strcasecmp(key, "Esc") == 0 || SDL_strcasecmp(key, "Escape") == 0) special = 0x1b;
  else if (SDL_strcasecmp(key, "Left") == 0) special = NSLeftArrowFunctionKey;
  else if (SDL_strcasecmp(key, "Right") == 0) special = NSRightArrowFunctionKey;
  else if (SDL_strcasecmp(key, "Up") == 0) special = NSUpArrowFunctionKey;
  else if (SDL_strcasecmp(key, "Down") == 0) special = NSDownArrowFunctionKey;
  else if (SDL_strcasecmp(key, "Home") == 0) special = NSHomeFunctionKey;
  else if (SDL_strcasecmp(key, "End") == 0) special = NSEndFunctionKey;
  else if (SDL_strcasecmp(key, "PageUp") == 0) special = NSPageUpFunctionKey;
  else if (SDL_strcasecmp(key, "PageDown") == 0) special = NSPageDownFunctionKey;
  else if (SDL_strcasecmp(key, "Yen") == 0) special = 0x00a5;
  if (special) return [NSString stringWithCharacters:&special length:1];
  (void)mask_io;
  return @"";
}

static NSEventModifierFlags mask_of(int32_t mods) {
  NSEventModifierFlags m = 0;
  if (mods & MOD_SHIFT) m |= NSEventModifierFlagShift;
  if (mods & MOD_CTRL)  m |= NSEventModifierFlagControl;
  if (mods & MOD_ALT)   m |= NSEventModifierFlagOption;
  if (mods & MOD_META)  m |= NSEventModifierFlagCommand;
  return m;
}

// flags: bit 0 = enabled, bit 1 = checked, bit 2..5 = mods。packed は "id\tlabel\tkey"。
void um_mac_menu_item(int32_t flags, const char *packed) {
  @autoreleasepool {
    if (staged_stack.count <= 1 || !packed) return;
    const char *t1 = strchr(packed, '\t');
    if (!t1) return;
    const char *t2 = strchr(t1 + 1, '\t');
    if (!t2) return;
    NSString *cmd = [[NSString alloc] initWithBytes:packed length:(NSUInteger)(t1 - packed) encoding:NSUTF8StringEncoding];
    NSString *label = [[NSString alloc] initWithBytes:t1 + 1 length:(NSUInteger)(t2 - (t1 + 1)) encoding:NSUTF8StringEncoding];
    const char *key = t2 + 1;
    if (!cmd || !label) return;
    if (!menu_target) menu_target = [[UMMenuTarget alloc] init];
    NSEventModifierFlags mask = mask_of((flags >> 2) & 15);
    NSString *ke = key_equiv_of(key, &mask);
    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:label action:@selector(onMenuItem:) keyEquivalent:ke];
    item.keyEquivalentModifierMask = ke.length > 0 ? mask : 0;
    item.target = menu_target;
    item.representedObject = cmd;
    item.tag = (flags & 1) ? 1 : 0;
    item.state = (flags & 2) ? NSControlStateValueOn : NSControlStateValueOff;
    [staged_stack.lastObject addItem:item];
    if (cmd.length > 0) staged_items[cmd] = item;
  }
}

void um_mac_menu_role(int32_t role, const char *label) {
  @autoreleasepool {
    if (staged_stack.count <= 1) return;
    NSMenu *menu = staged_stack.lastObject;
    NSString *t = ns_str(label);
    NSMenuItem *item = nil;
    switch (role) {
      case ROLE_SERVICES: {
        item = [[NSMenuItem alloc] initWithTitle:t action:nil keyEquivalent:@""];
        NSMenu *sv = [[NSMenu alloc] initWithTitle:t];
        item.submenu = sv;
        staged_services = sv;
        break;
      }
      case ROLE_HIDE:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(hide:) keyEquivalent:@"h"];
        item.keyEquivalentModifierMask = NSEventModifierFlagCommand;
        item.target = NSApp;
        break;
      case ROLE_HIDE_OTHERS:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
        item.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagOption;
        item.target = NSApp;
        break;
      case ROLE_SHOW_ALL:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(unhideAllApplications:) keyEquivalent:@""];
        item.target = NSApp;
        break;
      case ROLE_MINIMIZE:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(performMiniaturize:) keyEquivalent:@"m"];
        item.keyEquivalentModifierMask = NSEventModifierFlagCommand;
        break;
      case ROLE_ZOOM:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(performZoom:) keyEquivalent:@""];
        break;
      case ROLE_FULLSCREEN:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(toggleFullScreen:) keyEquivalent:@"f"];
        item.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagControl;
        break;
      case ROLE_BRING_ALL_FRONT:
        item = [[NSMenuItem alloc] initWithTitle:t action:@selector(arrangeInFront:) keyEquivalent:@""];
        item.target = NSApp;
        break;
      default:
        break;
    }
    if (item) [menu addItem:item];
  }
}

int32_t um_mac_menu_commit(Uint32 event_type) {
  @autoreleasepool {
    if (!staged_bar) return 0;
    menu_event_type = event_type;
    if (!um_mac_menu_has_native() || !NSApp) {
      clear_staged_menu();
      return 0;
    }
    if (staged_services) NSApp.servicesMenu = staged_services;
    if (staged_windows) NSApp.windowsMenu = staged_windows;
    [NSApp setMainMenu:staged_bar];
    live_items = staged_items;
    clear_staged_menu();
    return 1;
  }
}

int32_t um_mac_menu_set_state(int32_t flags, const char *id) {
  @autoreleasepool {
    if (!live_items || !id) return 0;
    NSMenuItem *item = live_items[ns_str(id)];
    if (!item) return 0;
    item.tag = (flags & 1) ? 1 : 0;
    item.state = (flags & 2) ? NSControlStateValueOn : NSControlStateValueOff;
    return 1;
  }
}
